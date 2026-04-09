const express = require('express');
const https = require('https');
const http = require('http');
const { WebSocketServer, WebSocket } = require('ws');
const multer = require('multer');
const fs = require('fs');
const FormData = require('form-data');
const crypto = require('crypto');
require('dotenv').config();

const app = express();
app.use(express.json());

// CORS — permite peticiones desde GitHub Pages y cualquier origen
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

const upload = multer({ dest: 'uploads/' });

// ── Helper: HTTPS POST ────────────────────────────────────────────
function httpsPost(hostname, path, headers, body) {
  return new Promise((resolve, reject) => {
    const data = typeof body === 'string' ? body : JSON.stringify(body);
    const options = {
      hostname, path, method: 'POST',
      headers: { ...headers, 'Content-Length': Buffer.byteLength(data) },
    };
    const req = https.request(options, (res) => {
      let raw = '';
      res.on('data', chunk => raw += chunk);
      res.on('end', () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(raw) }); }
        catch { resolve({ status: res.statusCode, body: raw }); }
      });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

// ── POST /translate ───────────────────────────────────────────────
app.post('/translate', async (req, res) => {
  const { text, target_lang, context } = req.body;
  if (!text || !target_lang) return res.status(400).json({ error: 'MISSING_PARAMS' });
  try {
    const params = new URLSearchParams({ text, target_lang });
    if (context) params.append('context', context);
    const result = await httpsPost(
      'api-free.deepl.com', '/v2/translate',
      { 'Authorization': `DeepL-Auth-Key ${process.env.DEEPL_API_KEY}`, 'Content-Type': 'application/x-www-form-urlencoded' },
      params.toString()
    );
    if (result.status !== 200) return res.status(502).json({ error: 'DEEPL_ERROR', detail: result.body });
    const translation = result.body.translations[0];
    return res.json({ translatedText: translation.text, detectedLang: translation.detected_source_language, engine: 'deepl' });
  } catch (e) {
    return res.status(502).json({ error: 'DEEPL_EXCEPTION', message: e.message });
  }
});

// ── Filtro de alucinaciones Whisper ───────────────────────────────
const NO_SPEECH_THRESHOLD = 0.6;
const AVG_LOGPROB_THRESHOLD = -0.8;
const HALLUCINATION_PHRASES = [
  'suscríbete', 'subscribe', 'like y suscríbete', 'gracias por ver',
  'thanks for watching', 'thank you for watching', 'don\'t forget to',
  'no olvides', 'síguenos', 'follow us', 'visit our website',
  'www.', '.com', '.es', 'youtube', 'instagram', 'twitter',
];

function isHallucination(text, segments) {
  if (!text || !text.trim()) return true;
  const lower = text.toLowerCase().trim();
  if (HALLUCINATION_PHRASES.some(phrase => lower.includes(phrase))) {
    console.log(`[FILTER] Hallucination phrase: "${text}"`);
    return true;
  }
  if (segments && segments.length > 0) {
    const avgNoSpeech = segments.reduce((sum, s) => sum + (s.no_speech_prob || 0), 0) / segments.length;
    const avgLogprob = segments.reduce((sum, s) => sum + (s.avg_logprob || 0), 0) / segments.length;
    if (avgNoSpeech > NO_SPEECH_THRESHOLD) { console.log(`[FILTER] no_speech_prob: ${avgNoSpeech.toFixed(2)}`); return true; }
    if (avgLogprob < AVG_LOGPROB_THRESHOLD) { console.log(`[FILTER] avg_logprob: ${avgLogprob.toFixed(2)}`); return true; }
  }
  return false;
}

// ── POST /transcribe ──────────────────────────────────────────────
app.post('/transcribe', upload.single('audio'), async (req, res) => {
  const language = req.body.language ?? 'es';
  const filePath = req.file?.path;
  if (!filePath) return res.status(400).json({ error: 'NO_AUDIO', text: '' });
  try {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath), { filename: 'audio.m4a', contentType: 'audio/m4a' });
    formData.append('model', 'whisper-large-v3-turbo');
    formData.append('language', language.slice(0, 2).toLowerCase());
    formData.append('response_format', 'verbose_json');
    const groqRes = await new Promise((resolve, reject) => {
      const options = {
        hostname: 'api.groq.com', path: '/openai/v1/audio/transcriptions', method: 'POST',
        headers: { 'Authorization': `Bearer ${process.env.GROQ_API_KEY}`, ...formData.getHeaders() },
      };
      const reqGroq = https.request(options, (response) => {
        let raw = '';
        response.on('data', chunk => raw += chunk);
        response.on('end', () => {
          try { resolve({ status: response.statusCode, body: JSON.parse(raw) }); }
          catch { resolve({ status: response.statusCode, body: raw }); }
        });
      });
      reqGroq.on('error', reject);
      formData.pipe(reqGroq);
    });
    fs.unlink(filePath, () => {});
    if (groqRes.status !== 200) { console.error('Groq error:', groqRes.status, groqRes.body); return res.status(502).json({ error: 'GROQ_ERROR', text: '' }); }
    const text = groqRes.body.text ?? '';
    const segments = groqRes.body.segments ?? [];
    if (isHallucination(text, segments)) return res.json({ text: '', engine: 'groq-whisper', filtered: true });
    console.log(`Transcribed: "${text.slice(0, 80)}"`);
    return res.json({ text, engine: 'groq-whisper' });
  } catch (e) {
    console.error('Transcribe exception:', e.message);
    if (filePath && fs.existsSync(filePath)) fs.unlink(filePath, () => {});
    return res.status(500).json({ error: 'TRANSCRIBE_EXCEPTION', text: '' });
  }
});

// ── POST /transcribe-realtime ─────────────────────────────────────
// Igual que /transcribe pero pensado para el modo conversación en tiempo real
// Devuelve el texto transcrito para que el backend haga la traducción via WS
app.post('/transcribe-realtime', upload.single('audio'), async (req, res) => {
  const language = req.body.language ?? 'es';
  const filePath = req.file?.path;
  if (!filePath) return res.status(400).json({ error: 'NO_AUDIO', text: '' });
  try {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath), { filename: 'audio.m4a', contentType: 'audio/m4a' });
    formData.append('model', 'whisper-large-v3-turbo');
    formData.append('language', language.slice(0, 2).toLowerCase());
    formData.append('response_format', 'verbose_json');
    const groqRes = await new Promise((resolve, reject) => {
      const options = {
        hostname: 'api.groq.com', path: '/openai/v1/audio/transcriptions', method: 'POST',
        headers: { 'Authorization': `Bearer ${process.env.GROQ_API_KEY}`, ...formData.getHeaders() },
      };
      const reqGroq = https.request(options, (response) => {
        let raw = '';
        response.on('data', chunk => raw += chunk);
        response.on('end', () => {
          try { resolve({ status: response.statusCode, body: JSON.parse(raw) }); }
          catch { resolve({ status: response.statusCode, body: raw }); }
        });
      });
      reqGroq.on('error', reject);
      formData.pipe(reqGroq);
    });
    fs.unlink(filePath, () => {});
    if (groqRes.status !== 200) return res.status(502).json({ error: 'GROQ_ERROR', text: '' });
    const text = groqRes.body.text ?? '';
    const segments = groqRes.body.segments ?? [];
    if (isHallucination(text, segments)) return res.json({ text: '', filtered: true });
    return res.json({ text });
  } catch (e) {
    if (filePath && fs.existsSync(filePath)) fs.unlink(filePath, () => {});
    return res.status(500).json({ error: 'TRANSCRIBE_EXCEPTION', text: '' });
  }
});

// ── GESTIÓN DE SALAS ──────────────────────────────────────────────
// rooms: Map<roomId, { host, guest, langHost, langGuest, lastActivity, timeout }>
const rooms = new Map();
const ROOM_INACTIVITY_MS = 5 * 60 * 1000; // 5 minutos

function generateRoomId() {
  return String(Math.floor(10000 + Math.random() * 90000)); // 5 dígitos: 10000-99999
}

function sendToWs(ws, data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(data));
  }
}

function closeRoom(roomId, reason = 'closed') {
  const room = rooms.get(roomId);
  if (!room) return;
  clearTimeout(room.timeout);
  sendToWs(room.host, { type: 'room_closed', reason });
  sendToWs(room.guest, { type: 'room_closed', reason });
  rooms.delete(roomId);
  console.log(`[ROOM] ${roomId} closed — ${reason}`);
}

function resetInactivityTimer(roomId) {
  const room = rooms.get(roomId);
  if (!room) return;
  clearTimeout(room.timeout);
  room.lastActivity = Date.now();
  room.timeout = setTimeout(() => closeRoom(roomId, 'inactivity'), ROOM_INACTIVITY_MS);
}

// ── POST /room/create ─────────────────────────────────────────────
app.post('/room/create', (req, res) => {
  const { langHost, langGuest } = req.body;
  if (!langHost || !langGuest) return res.status(400).json({ error: 'MISSING_LANGS' });

  const roomId = generateRoomId();
  rooms.set(roomId, {
    host: null, guest: null,
    langHost, langGuest,
    lastActivity: Date.now(),
    timeout: setTimeout(() => closeRoom(roomId, 'inactivity'), ROOM_INACTIVITY_MS),
  });

  console.log(`[ROOM] Created ${roomId} — host:${langHost} guest:${langGuest}`);
  return res.json({ roomId, langHost, langGuest });
});

// ── GET /room/:roomId ─────────────────────────────────────────────
app.get('/room/:roomId', (req, res) => {
  const room = rooms.get(req.params.roomId);
  if (!room) return res.status(404).json({ error: 'ROOM_NOT_FOUND' });
  return res.json({
    roomId: req.params.roomId,
    langHost: room.langHost,
    langGuest: room.langGuest,
    hasHost: !!room.host,
    hasGuest: !!room.guest,
  });
});

// ── POST /room/translate-and-relay ───────────────────────────────
// El cliente envía audio transcrito + roomId + role
// El backend traduce y lo reenvía al otro participante via WS
app.post('/room/relay', async (req, res) => {
  const { roomId, role, text } = req.body; // role: 'host' | 'guest'
  if (!roomId || !role || !text) return res.status(400).json({ error: 'MISSING_PARAMS' });

  const room = rooms.get(roomId);
  if (!room) return res.status(404).json({ error: 'ROOM_NOT_FOUND' });

  resetInactivityTimer(roomId);

  try {
    // Determinar idiomas según el rol
    const srcLang = role === 'host' ? room.langHost : room.langGuest;
    const tgtLang = role === 'host' ? room.langGuest : room.langHost;

    // Traducir con DeepL
    const params = new URLSearchParams({ text, target_lang: tgtLang });
    const result = await httpsPost(
      'api-free.deepl.com', '/v2/translate',
      { 'Authorization': `DeepL-Auth-Key ${process.env.DEEPL_API_KEY}`, 'Content-Type': 'application/x-www-form-urlencoded' },
      params.toString()
    );

    if (result.status !== 200) return res.status(502).json({ error: 'DEEPL_ERROR' });
    const translatedText = result.body.translations[0].text;

    // Enviar al otro participante via WebSocket
    const targetWs = role === 'host' ? room.guest : room.host;
    sendToWs(targetWs, {
      type: 'translation',
      from: role,
      original: text,
      translated: translatedText,
      srcLang,
      tgtLang,
    });

    // También confirmar al emisor
    return res.json({ ok: true, translatedText, relayed: !!targetWs });
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
});

// ── GET /join/:roomId ────────────────────────────────────────────
// Serves the web guest page for real-time conversation
const GUEST_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <title>Parlora AI</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0D0D14;
      --card: rgba(255,255,255,0.05);
      --border: rgba(255,255,255,0.09);
      --purple: #818CF8;
      --purple-light: #C4B5FD;
      --green: #34C759;
      --red: #EF4444;
      --text: #fff;
      --text-dim: rgba(255,255,255,0.45);
      --text-faint: rgba(255,255,255,0.25);
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    /* ── SCREENS ─────────────────────────────────────── */
    .screen { display: none; flex-direction: column; align-items: center; width: 100%; max-width: 480px; min-height: 100vh; padding: 24px 20px; }
    .screen.active { display: flex; }

    /* ── LOGO ─────────────────────────────────────────── */
    .logo-wrap { width: 72px; height: 72px; border-radius: 20px; background: rgba(99,102,241,0.15); border: 0.5px solid rgba(99,102,241,0.3); display: flex; align-items: center; justify-content: center; font-size: 32px; margin-bottom: 16px; margin-top: 32px; }
    .app-name { font-size: 28px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 6px; }
    .tagline { font-size: 14px; color: var(--text-dim); text-align: center; line-height: 1.5; margin-bottom: 32px; }

    /* ── CARDS ────────────────────────────────────────── */
    .card { background: var(--card); border: 0.5px solid var(--border); border-radius: 18px; padding: 18px; width: 100%; margin-bottom: 14px; }
    .card-label { font-size: 10px; font-weight: 700; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; }
    .card-value { font-size: 15px; font-weight: 600; color: var(--text); }

    /* ── LANG PILLS ───────────────────────────────────── */
    .lang-scroll { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 4px; scrollbar-width: none; }
    .lang-scroll::-webkit-scrollbar { display: none; }
    .lang-pill { flex-shrink: 0; padding: 7px 14px; border-radius: 20px; border: 0.5px solid var(--border); background: rgba(255,255,255,0.04); font-size: 13px; color: var(--text-dim); cursor: pointer; white-space: nowrap; transition: all 0.15s; }
    .lang-pill.active-a { border-color: var(--purple); background: rgba(129,140,248,0.15); color: var(--purple); }
    .lang-pill.active-b { border-color: var(--purple-light); background: rgba(196,181,253,0.15); color: var(--purple-light); }
    .lang-pill:hover { opacity: 0.8; }

    /* ── BUTTONS ──────────────────────────────────────── */
    .btn { width: 100%; padding: 15px; border-radius: 14px; border: none; font-size: 15px; font-weight: 700; cursor: pointer; transition: opacity 0.15s; }
    .btn:hover { opacity: 0.85; }
    .btn-primary { background: var(--purple); color: #fff; }
    .btn-secondary { background: rgba(255,255,255,0.06); border: 0.5px solid var(--border); color: var(--text); }
    .btn-danger { background: rgba(239,68,68,0.12); border: 0.5px solid rgba(239,68,68,0.3); color: var(--red); }
    .btn-green { background: rgba(52,199,89,0.12); border: 0.5px solid rgba(52,199,89,0.3); color: var(--green); }
    .btn:disabled { opacity: 0.35; cursor: not-allowed; }

    /* ── STATUS BADGE ─────────────────────────────────── */
    .status-badge { display: flex; align-items: center; gap: 7px; padding: 5px 14px; border-radius: 20px; border: 0.5px solid var(--border); font-size: 12px; font-weight: 500; color: var(--text-dim); }
    .status-badge.active { border-color: rgba(52,199,89,0.3); background: rgba(52,199,89,0.08); color: var(--green); }
    .status-badge.connecting { border-color: rgba(251,191,36,0.3); background: rgba(251,191,36,0.08); color: #FBbf24; }
    .status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text-faint); }
    .status-badge.active .status-dot { background: var(--green); }
    .status-badge.connecting .status-dot { background: #FBbf24; }

    /* ── HEADER ───────────────────────────────────────── */
    .session-header { display: flex; align-items: center; justify-content: space-between; width: 100%; padding: 12px 0 8px; }

    /* ── CHAT BUBBLES ─────────────────────────────────── */
    .chat-box { flex: 1; width: 100%; overflow-y: auto; padding: 8px 0; min-height: 200px; }
    .bubble-row { display: flex; align-items: flex-end; gap: 8px; margin-bottom: 12px; }
    .bubble-row.from-host { flex-direction: row; }
    .bubble-row.from-guest { flex-direction: row-reverse; }
    .bubble-avatar { width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; flex-shrink: 0; margin-bottom: 2px; }
    .bubble { max-width: 75%; border-radius: 18px; padding: 11px 14px; }
    .bubble-row.from-host .bubble { background: rgba(129,140,248,0.15); border-bottom-left-radius: 4px; }
    .bubble-row.from-guest .bubble { background: rgba(196,181,253,0.15); border-bottom-right-radius: 4px; }
    .bubble-name { font-size: 10px; font-weight: 700; margin-bottom: 4px; }
    .bubble-row.from-host .bubble-name { color: var(--purple); }
    .bubble-row.from-guest .bubble-name { color: var(--purple-light); }
    .bubble-original { font-size: 13px; color: rgba(255,255,255,0.8); line-height: 1.4; }
    .bubble-divider { height: 0.5px; background: var(--border); margin: 6px 0; }
    .bubble-translated { font-size: 12px; color: rgba(255,255,255,0.5); font-style: italic; line-height: 1.4; }
    .bubble-repeat { background: none; border: none; cursor: pointer; float: right; margin-top: 4px; font-size: 12px; opacity: 0.6; }
    .bubble-repeat:hover { opacity: 1; }

    /* ── RECORDING INDICATOR ──────────────────────────── */
    .recording-bar { display: flex; align-items: center; gap: 10px; background: var(--card); border-radius: 12px; padding: 10px 14px; width: 100%; margin-bottom: 8px; }
    .recording-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
    .recording-text { font-size: 11px; color: var(--text-dim); flex: 1; }
    .audio-level { height: 3px; background: rgba(255,255,255,0.08); border-radius: 2px; overflow: hidden; margin-top: 4px; }
    .audio-level-fill { height: 3px; background: var(--purple); border-radius: 2px; transition: width 0.1s; width: 0%; }

    /* ── SPEED SELECTOR ───────────────────────────────── */
    .speed-row { display: flex; align-items: center; gap: 6px; width: 100%; margin-bottom: 8px; }
    .speed-label { font-size: 11px; color: var(--text-faint); margin-right: 4px; }
    .speed-btn { padding: 5px 10px; border-radius: 12px; border: 0.5px solid var(--border); background: rgba(255,255,255,0.04); font-size: 11px; color: var(--text-dim); cursor: pointer; font-weight: 600; }
    .speed-btn.active { border-color: var(--purple); background: rgba(129,140,248,0.15); color: var(--purple); }

    /* ── WARNING ──────────────────────────────────────── */
    .warning-box { background: rgba(239,68,68,0.08); border: 0.5px solid rgba(239,68,68,0.3); border-radius: 14px; padding: 14px; width: 100%; margin-bottom: 14px; font-size: 13px; color: rgba(255,255,255,0.5); line-height: 1.5; }
    .info-box { background: rgba(129,140,248,0.08); border: 0.5px solid rgba(129,140,248,0.25); border-radius: 14px; padding: 14px; width: 100%; margin-bottom: 14px; font-size: 13px; color: rgba(255,255,255,0.6); line-height: 1.5; }

    /* ── EMPTY STATE ──────────────────────────────────── */
    .empty-text { font-size: 12px; color: var(--text-faint); text-align: center; padding: 30px 0; line-height: 1.7; }

    /* ── HINT ─────────────────────────────────────────── */
    .hint { font-size: 11px; color: var(--text-faint); text-align: center; padding: 10px 20px; }

    /* ── LANG INFO ROW ────────────────────────────────── */
    .lang-info-row { display: flex; align-items: center; gap: 8px; width: 100%; margin-bottom: 14px; }
    .lang-info-card { flex: 1; background: var(--card); border: 1.5px solid var(--border); border-radius: 16px; padding: 12px; text-align: center; }
    .lang-info-icon { font-size: 20px; margin-bottom: 4px; }
    .lang-info-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
    .lang-info-name { font-size: 12px; color: rgba(255,255,255,0.6); }
    .lang-arrow { font-size: 18px; color: var(--text-faint); }

    /* ── SPINNER ──────────────────────────────────────── */
    .spinner { width: 40px; height: 40px; border: 3px solid var(--border); border-top-color: var(--purple); border-radius: 50%; animation: spin 0.8s linear infinite; margin: 20px auto; }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── FOOTER ───────────────────────────────────────── */
    .footer { font-size: 11px; color: var(--text-faint); text-align: center; padding: 20px 0 12px; }
    .footer a { color: rgba(99,102,241,0.7); text-decoration: none; }

    /* ── DIVIDER ──────────────────────────────────────── */
    .section-label { font-size: 10px; font-weight: 600; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.5px; width: 100%; margin-bottom: 6px; margin-top: 4px; }

    /* ── NOTIFICATION ─────────────────────────────────── */
    .notif { position: fixed; top: 16px; left: 50%; transform: translateX(-50%); background: #1A1A2E; border: 0.5px solid var(--purple); border-radius: 12px; padding: 12px 20px; font-size: 13px; color: var(--text); z-index: 999; max-width: 340px; text-align: center; box-shadow: 0 8px 24px rgba(0,0,0,0.4); display: none; }
    .notif.show { display: block; }
  </style>
</head>
<body>

<!-- ── SCREEN: LOADING ───────────────────────────────────────── -->
<div id="screen-loading" class="screen active">
  <div class="logo-wrap">🌐</div>
  <div class="app-name">Parlora AI</div>
  <div class="spinner"></div>
  <div class="tagline" id="loading-text">Connecting to room...</div>
</div>

<!-- ── SCREEN: CONFIRM LANGS ────────────────────────────────── -->
<div id="screen-confirm" class="screen">
  <div class="logo-wrap">🌐</div>
  <div class="app-name">Parlora AI</div>
  <div class="tagline">Real-time simultaneous translation</div>

  <div class="info-box" id="anc-warning">
    🎧 <strong>For best quality</strong> use noise-cancelling earphones (ANC). Without them, your mic may pick up the translation audio.
  </div>

  <div class="card">
    <div class="card-label">You will speak in</div>
    <div class="lang-scroll" id="guest-speak-langs"></div>
  </div>

  <div class="card">
    <div class="card-label">You will hear the translation in</div>
    <div class="lang-scroll" id="guest-hear-langs"></div>
  </div>

  <div class="warning-box" id="lang-warning" style="display:none">
    ⚠️ You selected the same language for speaking and hearing. Please choose different languages.
  </div>

  <button class="btn btn-primary" id="btn-join" style="margin-top:8px">Join session →</button>

  <div class="footer">
    Powered by <a href="https://parlora.ai" target="_blank">Parlora AI</a>
  </div>
</div>

<!-- ── SCREEN: SESSION ───────────────────────────────────────── -->
<div id="screen-session" class="screen">
  <div class="session-header">
    <button class="btn btn-secondary" id="btn-leave" style="width:auto;padding:8px 16px;font-size:13px">Leave</button>
    <div class="status-badge connecting" id="status-badge">
      <div class="status-dot"></div>
      <span id="status-text">Connecting...</span>
    </div>
    <button class="btn btn-secondary" id="btn-repeat" style="width:auto;padding:8px 12px;font-size:16px;opacity:0.4" disabled>🔁</button>
  </div>

  <div class="lang-info-row" id="lang-info-row">
    <!-- filled by JS -->
  </div>

  <div class="recording-bar" id="recording-bar" style="display:none">
    <div class="recording-dot" id="recording-dot" style="background:#EF4444"></div>
    <div style="flex:1">
      <div class="recording-text" id="recording-text">Capturing audio...</div>
      <div class="audio-level"><div class="audio-level-fill" id="audio-level-fill"></div></div>
    </div>
  </div>

  <div class="speed-row">
    <span class="speed-label">Speed:</span>
    <button class="speed-btn" data-speed="0.5">0.5x</button>
    <button class="speed-btn" data-speed="0.75">0.75x</button>
    <button class="speed-btn active" data-speed="1">1x</button>
    <button class="speed-btn" data-speed="1.25">1.25x</button>
    <button class="speed-btn" data-speed="1.5">1.5x</button>
  </div>

  <div class="section-label">Live translation</div>
  <div class="chat-box" id="chat-box">
    <div class="empty-text" id="empty-text">
      Waiting for the session to start...<br>Make sure both participants are connected.
    </div>
  </div>

  <div class="hint" id="session-hint">🎧 Keep earphones on — translation will play automatically</div>

  <button class="btn btn-danger" id="btn-stop" style="margin-top:8px;display:none">⏹ Stop session</button>
  <button class="btn btn-green" id="btn-start" style="margin-top:8px">▶ Start</button>
</div>

<!-- ── SCREEN: ENDED ─────────────────────────────────────────── -->
<div id="screen-ended" class="screen">
  <div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;text-align:center">
    <div style="font-size:56px">👋</div>
    <div style="font-size:22px;font-weight:700">Session ended</div>
    <div style="font-size:14px;color:var(--text-dim)" id="ended-reason">The session has been closed.</div>
  </div>
</div>

<!-- ── NOTIFICATION ───────────────────────────────────────────── -->
<div class="notif" id="notif"></div>

<script>
// ── CONFIG ────────────────────────────────────────────────────────
const BACKEND_HTTP = '';
const BACKEND_WS   = \`wss://\${location.host}/ws\`;
const CHUNK_MS = 4000; // 4 segundos por chunk

// ── LANGUAGES ────────────────────────────────────────────────────
const LANGUAGES = [
  { code: 'ES', name: 'Español',    flag: '🇪🇸' },
  { code: 'EN', name: 'English',    flag: '🇬🇧' },
  { code: 'FR', name: 'Français',   flag: '🇫🇷' },
  { code: 'DE', name: 'Deutsch',    flag: '🇩🇪' },
  { code: 'IT', name: 'Italiano',   flag: '🇮🇹' },
  { code: 'PT', name: 'Português',  flag: '🇵🇹' },
  { code: 'SK', name: 'Slovenčina', flag: '🇸🇰' },
  { code: 'CS', name: 'Čeština',    flag: '🇨🇿' },
  { code: 'PL', name: 'Polski',     flag: '🇵🇱' },
  { code: 'NL', name: 'Nederlands', flag: '🇳🇱' },
];

// ── STATE ─────────────────────────────────────────────────────────
let roomId = null;
let roomData = null;
let guestSpeakLang = null; // idioma que habla el guest
let guestHearLang = null;  // idioma que escucha el guest (= lo que habla el host)
let ws = null;
let isActive = false;
let mediaRecorder = null;
let audioContext = null;
let analyser = null;
let animFrameId = null;
let ttsSpeed = 1.0;
let lastTranslation = null;
let chunkCount = 0;
let pingInterval = null;

// ── HELPERS ───────────────────────────────────────────────────────
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById('screen-' + id).classList.add('active');
}

function showNotif(msg, duration = 3000) {
  const el = document.getElementById('notif');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), duration);
}

function getLang(code) { return LANGUAGES.find(l => l.code === code) || { code, name: code, flag: '🌐' }; }

function setStatus(text, state = '') {
  const badge = document.getElementById('status-badge');
  const span = document.getElementById('status-text');
  badge.className = 'status-badge' + (state ? ' ' + state : '');
  span.textContent = text;
}

function addBubble(from, original, translated, srcLang, tgtLang) {
  const box = document.getElementById('chat-box');
  const empty = document.getElementById('empty-text');
  if (empty) empty.remove();

  const srcL = getLang(srcLang);
  const tgtL = getLang(tgtLang);
  const isHost = from === 'host';
  const name = isHost ? 'Host' : 'You';
  const translId = 'tl-' + Date.now();

  const row = document.createElement('div');
  row.className = 'bubble-row ' + (isHost ? 'from-host' : 'from-guest');
  row.innerHTML = \`
    <div class="bubble-avatar" style="background:\${isHost ? 'rgba(129,140,248,0.2)' : 'rgba(196,181,253,0.2)'}">
      \${name.charAt(0)}
    </div>
    <div class="bubble">
      <div class="bubble-name">\${name}</div>
      <div class="bubble-original">\${srcL.flag} \${escHtml(original)}</div>
      <div class="bubble-divider"></div>
      <div class="bubble-translated" id="\${translId}">\${tgtL.flag} \${escHtml(translated)}</div>
      <button class="bubble-repeat" onclick="replayTts('\${escHtml(translated)}','\${tgtLang}')">🔁</button>
    </div>
  \`;
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;

  lastTranslation = { translated, ttsLocale: tgtLang };
  const repeatBtn = document.getElementById('btn-repeat');
  if (repeatBtn) { repeatBtn.disabled = false; repeatBtn.style.opacity = '1'; }
}

function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

function replayTts(text, lang) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.lang = langCodeToLocale(lang);
  utt.rate = ttsSpeed;
  window.speechSynthesis.speak(utt);
}

function langCodeToLocale(code) {
  const map = { ES:'es-ES', EN:'en-US', FR:'fr-FR', DE:'de-DE', IT:'it-IT', PT:'pt-PT', SK:'sk-SK', CS:'cs-CZ', PL:'pl-PL', NL:'nl-NL' };
  return map[code] || code.toLowerCase();
}

// ── INIT ──────────────────────────────────────────────────────────
async function init() {
  // Leer roomId de la URL: ?room=XXXX
  const params = new URLSearchParams(window.location.search);
  roomId = params.get('room');

  if (!roomId) {
    document.getElementById('loading-text').textContent = 'Invalid link. Please scan the QR code again.';
    return;
  }

  // Fetch room info
  try {
    const r = await fetch(\`\${BACKEND_HTTP}/room/\${roomId}\`);
    if (!r.ok) throw new Error('Room not found');
    roomData = await r.json();
  } catch (e) {
    document.getElementById('loading-text').textContent = 'Room not found or expired. Please ask the host to create a new session.';
    return;
  }

  // Por defecto: el guest habla en langGuest y escucha langHost
  guestSpeakLang = roomData.langGuest;
  guestHearLang = roomData.langHost;

  buildConfirmScreen();
  showScreen('confirm');
}

function buildConfirmScreen() {
  // Selector idioma hablar (por defecto langGuest del host)
  const speakContainer = document.getElementById('guest-speak-langs');
  const hearContainer = document.getElementById('guest-hear-langs');

  LANGUAGES.forEach(l => {
    const pill1 = document.createElement('button');
    pill1.className = 'lang-pill' + (l.code === guestSpeakLang ? ' active-b' : '');
    pill1.textContent = l.flag + ' ' + l.name;
    pill1.onclick = () => {
      guestSpeakLang = l.code;
      document.querySelectorAll('#guest-speak-langs .lang-pill').forEach(p => p.className = 'lang-pill');
      pill1.className = 'lang-pill active-b';
      validateLangs();
    };
    speakContainer.appendChild(pill1);

    const pill2 = document.createElement('button');
    pill2.className = 'lang-pill' + (l.code === guestHearLang ? ' active-a' : '');
    pill2.textContent = l.flag + ' ' + l.name;
    pill2.onclick = () => {
      guestHearLang = l.code;
      document.querySelectorAll('#guest-hear-langs .lang-pill').forEach(p => p.className = 'lang-pill');
      pill2.className = 'lang-pill active-a';
      validateLangs();
    };
    hearContainer.appendChild(pill2);
  });
}

function validateLangs() {
  const warn = document.getElementById('lang-warning');
  const btn = document.getElementById('btn-join');
  if (guestSpeakLang === guestHearLang) {
    warn.style.display = 'block';
    btn.disabled = true;
  } else {
    warn.style.display = 'none';
    btn.disabled = false;
  }
}

// ── JOIN SESSION ──────────────────────────────────────────────────
document.getElementById('btn-join').addEventListener('click', async () => {
  // Request mic permission
  try {
    await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    showNotif('❌ Microphone permission denied. Please allow access.');
    return;
  }
  buildSessionScreen();
  showScreen('session');
  connectWebSocket();
});

function buildSessionScreen() {
  const speakL = getLang(guestSpeakLang);
  const hearL = getLang(guestHearLang);
  const row = document.getElementById('lang-info-row');
  row.innerHTML = \`
    <div class="lang-info-card" style="border-color:rgba(196,181,253,0.3)">
      <div class="lang-info-icon">🎙</div>
      <div class="lang-info-label" style="color:var(--purple-light)">You speak</div>
      <div class="lang-info-name">\${speakL.flag} \${speakL.name}</div>
    </div>
    <div class="lang-arrow">→</div>
    <div class="lang-info-card" style="border-color:rgba(129,140,248,0.3)">
      <div class="lang-info-icon">🎧</div>
      <div class="lang-info-label" style="color:var(--purple)">You hear</div>
      <div class="lang-info-name">\${hearL.flag} \${hearL.name}</div>
    </div>
  \`;
}

// ── WEBSOCKET ─────────────────────────────────────────────────────
function connectWebSocket() {
  setStatus('Connecting...', 'connecting');
  ws = new WebSocket(\`\${BACKEND_WS}?roomId=\${roomId}&role=guest\`);

  ws.onopen = () => {
    console.log('[WS] Connected');
    pingInterval = setInterval(() => { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'ping' })); }, 25000);
  };

  ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      handleWsMessage(msg);
    } catch (err) { console.log('[WS] Parse error', err); }
  };

  ws.onclose = (e) => {
    clearInterval(pingInterval);
    if (e.code === 4004) {
      document.getElementById('ended-reason').textContent = 'Room not found or expired.';
      showScreen('ended');
    }
  };

  ws.onerror = () => setStatus('Connection error', '');
}

function handleWsMessage(msg) {
  switch (msg.type) {
    case 'connected':
      setStatus('Waiting for host...', 'connecting');
      break;
    case 'peer_joined':
      if (msg.role === 'host') {
        setStatus('Host connected — ready', 'active');
        showNotif('🎉 Host connected!');
      }
      break;
    case 'peer_left':
      if (msg.role === 'host') {
        setStatus('Host disconnected', '');
        showNotif('⚠️ Host left the session');
        stopRecording();
        setTimeout(() => {
          document.getElementById('ended-reason').textContent = 'The host has left the session.';
          showScreen('ended');
        }, 2000);
      }
      break;
    case 'translation':
      // Recibimos traducción del host → reproducir con TTS
      addBubble('host', msg.original, msg.translated, msg.srcLang, msg.tgtLang);
      if (isActive) {
        replayTts(msg.translated, msg.tgtLang);
      }
      break;
    case 'room_closed':
      stopRecording();
      document.getElementById('ended-reason').textContent = msg.reason === 'inactivity'
        ? 'Session closed due to inactivity (5 min).'
        : 'The session has been closed by the host.';
      showScreen('ended');
      break;
    case 'pong':
      break;
  }
}

// ── START / STOP ──────────────────────────────────────────────────
document.getElementById('btn-start').addEventListener('click', startSession);
document.getElementById('btn-stop').addEventListener('click', stopSession);

async function startSession() {
  const ok = await requestMic();
  if (!ok) return;
  isActive = true;
  chunkCount = 0;
  document.getElementById('btn-start').style.display = 'none';
  document.getElementById('btn-stop').style.display = 'block';
  document.getElementById('recording-bar').style.display = 'flex';
  setStatus('Active', 'active');
  recordChunk();
}

function stopSession() {
  isActive = false;
  stopRecording();
  document.getElementById('btn-start').style.display = 'block';
  document.getElementById('btn-stop').style.display = 'none';
  document.getElementById('recording-bar').style.display = 'none';
  setStatus('Paused', '');
}

async function requestMic() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    stream.getTracks().forEach(t => t.stop()); // solo verificar permiso
    return true;
  } catch (e) {
    showNotif('❌ Microphone permission denied.');
    return false;
  }
}

// ── RECORDING LOOP ────────────────────────────────────────────────
async function recordChunk() {
  if (!isActive) return;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

    // Nivel de audio visual
    audioContext = new AudioContext();
    analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    analyser.fftSize = 256;
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    function updateLevel() {
      if (!isActive) return;
      analyser.getByteFrequencyData(dataArray);
      const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
      const pct = Math.min(100, (avg / 128) * 100);
      document.getElementById('audio-level-fill').style.width = pct + '%';
      animFrameId = requestAnimationFrame(updateLevel);
    }
    updateLevel();

    // Actualizar indicador
    const dot = document.getElementById('recording-dot');
    const txt = document.getElementById('recording-text');
    dot.style.background = chunkCount === 0 ? '#EF4444' : '#34C759';
    txt.textContent = chunkCount === 0
      ? '🎙 Capturing audio — first translation in ~4s'
      : \`🎙 Recording · Translation active · \${chunkCount} chunks\`;

    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus' : 'audio/webm';

    mediaRecorder = new MediaRecorder(stream, { mimeType });
    const chunks = [];
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };

    mediaRecorder.onstop = async () => {
      // Stop audio level animation
      cancelAnimationFrame(animFrameId);
      document.getElementById('audio-level-fill').style.width = '0%';
      stream.getTracks().forEach(t => t.stop());
      if (audioContext) { audioContext.close(); audioContext = null; }

      if (!isActive) return;

      // Siguiente chunk inmediatamente
      setTimeout(() => recordChunk(), 0);

      // Procesar este chunk
      if (chunks.length > 0) {
        const blob = new Blob(chunks, { type: mimeType });
        await processChunk(blob);
      }
    };

    mediaRecorder.start();
    setTimeout(() => { if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop(); }, CHUNK_MS);

  } catch (e) {
    console.error('Record error:', e);
    if (isActive) setTimeout(() => recordChunk(), 500);
  }
}

function stopRecording() {
  isActive = false;
  cancelAnimationFrame(animFrameId);
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    try { mediaRecorder.stop(); } catch (e) {}
  }
  if (audioContext) { audioContext.close(); audioContext = null; }
  document.getElementById('audio-level-fill').style.width = '0%';
  document.getElementById('recording-bar').style.display = 'none';
}

async function processChunk(blob) {
  try {
    // 1. Transcribir
    const formData = new FormData();
    formData.append('audio', blob, 'chunk.webm');
    formData.append('language', guestSpeakLang.toLowerCase().slice(0,2));

    const transcribeRes = await fetch(\`\${BACKEND_HTTP}/transcribe-realtime\`, {
      method: 'POST', body: formData,
    });
    const { text = '' } = await transcribeRes.json();
    if (!text.trim()) return;

    // 2. Enviar al backend para traducir y relay al host
    const relayRes = await fetch(\`\${BACKEND_HTTP}/room/relay\`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ roomId, role: 'guest', text: text.trim() }),
    });
    const { translatedText = '', relayed } = await relayRes.json();
    if (!translatedText) return;

    chunkCount++;

    // 3. Mostrar burbuja propia
    addBubble('guest', text.trim(), translatedText, guestSpeakLang, guestHearLang);

    // Actualizar indicador
    document.getElementById('recording-dot').style.background = '#34C759';
    document.getElementById('recording-text').textContent =
      \`🎙 Recording · Translation active · \${chunkCount} chunks\`;

  } catch (e) {
    console.error('Process error:', e);
  }
}

// ── LEAVE / REPEAT ────────────────────────────────────────────────
document.getElementById('btn-leave').addEventListener('click', () => {
  stopRecording();
  if (ws) ws.close();
  document.getElementById('ended-reason').textContent = 'You left the session.';
  showScreen('ended');
});

document.getElementById('btn-repeat').addEventListener('click', () => {
  if (lastTranslation) replayTts(lastTranslation.translated, lastTranslation.ttsLocale);
});

// ── SPEED BUTTONS ─────────────────────────────────────────────────
document.querySelectorAll('.speed-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    ttsSpeed = parseFloat(btn.dataset.speed);
    document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
});

// ── START ─────────────────────────────────────────────────────────
init();
</script>
</body>
</html>
`;

app.get('/join/:roomId', (req, res) => {
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(GUEST_HTML);
});

app.get('/join', (req, res) => {
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(GUEST_HTML);
});

// ── GET / ─────────────────────────────────────────────────────────
app.get('/', (req, res) => res.json({ status: 'Parlora AI backend running ✓', rooms: rooms.size }));

// ── Servidor HTTP + WebSocket ─────────────────────────────────────
const PORT = process.env.PORT || 8080;
const server = http.createServer(app);

const wss = new WebSocketServer({ server, path: '/ws' });

wss.on('connection', (ws, req) => {
  const url = new URL(req.url, `http://localhost`);
  const roomId = url.searchParams.get('roomId');
  const role = url.searchParams.get('role'); // 'host' | 'guest'

  if (!roomId || !role) { ws.close(4000, 'Missing roomId or role'); return; }

  const room = rooms.get(roomId);
  if (!room) { ws.close(4004, 'Room not found'); return; }

  // Asignar cliente a la sala
  if (role === 'host') {
    if (room.host) { ws.close(4001, 'Host already connected'); return; }
    room.host = ws;
    console.log(`[WS] Host connected to room ${roomId}`);
    sendToWs(ws, { type: 'connected', role: 'host', roomId, langHost: room.langHost, langGuest: room.langGuest });
    if (room.guest) {
      sendToWs(room.guest, { type: 'peer_joined', role: 'host' });
      sendToWs(ws, { type: 'peer_joined', role: 'guest' });
    }
  } else if (role === 'guest') {
    if (room.guest) { ws.close(4002, 'Guest already connected'); return; }
    room.guest = ws;
    console.log(`[WS] Guest connected to room ${roomId}`);
    sendToWs(ws, { type: 'connected', role: 'guest', roomId, langHost: room.langHost, langGuest: room.langGuest, hasHost: !!room.host });
    if (room.host) {
      sendToWs(room.host, { type: 'peer_joined', role: 'guest' });
      sendToWs(ws, { type: 'peer_joined', role: 'host' });
    }
  } else {
    ws.close(4003, 'Invalid role');
    return;
  }

  resetInactivityTimer(roomId);

  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data.toString());
      resetInactivityTimer(roomId);

      // ping/pong keepalive
      if (msg.type === 'ping') { sendToWs(ws, { type: 'pong' }); return; }

      // El cliente puede enviar mensajes de texto directos (sin audio)
      if (msg.type === 'chat') {
        const targetWs = role === 'host' ? room.guest : room.host;
        sendToWs(targetWs, { type: 'chat', from: role, text: msg.text });
      }

      // Confirmación de que recibió la traducción (para métricas futuras)
      if (msg.type === 'ack') {
        console.log(`[ACK] Room ${roomId} role ${role} ack chunk ${msg.chunkId}`);
      }

    } catch (e) {
      console.log('[WS] Message parse error:', e.message);
    }
  });

  ws.on('close', () => {
    console.log(`[WS] ${role} disconnected from room ${roomId}`);
    const currentRoom = rooms.get(roomId);
    if (!currentRoom) return;

    // Notificar al otro participante
    const otherWs = role === 'host' ? currentRoom.guest : currentRoom.host;
    sendToWs(otherWs, { type: 'peer_left', role });

    // Limpiar referencia
    if (role === 'host') currentRoom.host = null;
    else currentRoom.guest = null;

    // Si los dos se fueron, cerrar sala
    if (!currentRoom.host && !currentRoom.guest) {
      closeRoom(roomId, 'all_left');
    }
  });

  ws.on('error', (e) => console.log(`[WS] Error room ${roomId} role ${role}:`, e.message));
});

server.listen(PORT, () => console.log(`Parlora AI backend escuchando en puerto ${PORT}`));
module.exports = app;