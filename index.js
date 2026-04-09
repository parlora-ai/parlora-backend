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

// ── GET / ─────────────────────────────────────────────────────────
const GUEST_HTML = Buffer.from('PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CiAgPG1ldGEgY2hhcnNldD0iVVRGLTgiIC8+CiAgPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwgaW5pdGlhbC1zY2FsZT0xLjAsIG1heGltdW0tc2NhbGU9MS4wLCB1c2VyLXNjYWxhYmxlPW5vIiAvPgogIDx0aXRsZT5QYXJsb3JhIEFJPC90aXRsZT4KICA8c3R5bGU+CiAgICAqIHsgYm94LXNpemluZzogYm9yZGVyLWJveDsgbWFyZ2luOiAwOyBwYWRkaW5nOiAwOyB9CgogICAgOnJvb3QgewogICAgICAtLWJnOiAjMEQwRDE0OwogICAgICAtLWNhcmQ6IHJnYmEoMjU1LDI1NSwyNTUsMC4wNSk7CiAgICAgIC0tYm9yZGVyOiByZ2JhKDI1NSwyNTUsMjU1LDAuMDkpOwogICAgICAtLXB1cnBsZTogIzgxOENGODsKICAgICAgLS1wdXJwbGUtbGlnaHQ6ICNDNEI1RkQ7CiAgICAgIC0tZ3JlZW46ICMzNEM3NTk7CiAgICAgIC0tcmVkOiAjRUY0NDQ0OwogICAgICAtLXRleHQ6ICNmZmY7CiAgICAgIC0tdGV4dC1kaW06IHJnYmEoMjU1LDI1NSwyNTUsMC40NSk7CiAgICAgIC0tdGV4dC1mYWludDogcmdiYSgyNTUsMjU1LDI1NSwwLjI1KTsKICAgIH0KCiAgICBib2R5IHsKICAgICAgYmFja2dyb3VuZDogdmFyKC0tYmcpOwogICAgICBjb2xvcjogdmFyKC0tdGV4dCk7CiAgICAgIGZvbnQtZmFtaWx5OiAtYXBwbGUtc3lzdGVtLCBCbGlua01hY1N5c3RlbUZvbnQsICdTZWdvZSBVSScsIHNhbnMtc2VyaWY7CiAgICAgIG1pbi1oZWlnaHQ6IDEwMHZoOwogICAgICBkaXNwbGF5OiBmbGV4OwogICAgICBmbGV4LWRpcmVjdGlvbjogY29sdW1uOwogICAgICBhbGlnbi1pdGVtczogY2VudGVyOwogICAgfQoKICAgIC8qIOKUgOKUgCBTQ1JFRU5TIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwogICAgLnNjcmVlbiB7IGRpc3BsYXk6IG5vbmU7IGZsZXgtZGlyZWN0aW9uOiBjb2x1bW47IGFsaWduLWl0ZW1zOiBjZW50ZXI7IHdpZHRoOiAxMDAlOyBtYXgtd2lkdGg6IDQ4MHB4OyBtaW4taGVpZ2h0OiAxMDB2aDsgcGFkZGluZzogMjRweCAyMHB4OyB9CiAgICAuc2NyZWVuLmFjdGl2ZSB7IGRpc3BsYXk6IGZsZXg7IH0KCiAgICAvKiDilIDilIAgTE9HTyDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAgKi8KICAgIC5sb2dvLXdyYXAgeyB3aWR0aDogNzJweDsgaGVpZ2h0OiA3MnB4OyBib3JkZXItcmFkaXVzOiAyMHB4OyBiYWNrZ3JvdW5kOiByZ2JhKDk5LDEwMiwyNDEsMC4xNSk7IGJvcmRlcjogMC41cHggc29saWQgcmdiYSg5OSwxMDIsMjQxLDAuMyk7IGRpc3BsYXk6IGZsZXg7IGFsaWduLWl0ZW1zOiBjZW50ZXI7IGp1c3RpZnktY29udGVudDogY2VudGVyOyBmb250LXNpemU6IDMycHg7IG1hcmdpbi1ib3R0b206IDE2cHg7IG1hcmdpbi10b3A6IDMycHg7IH0KICAgIC5hcHAtbmFtZSB7IGZvbnQtc2l6ZTogMjhweDsgZm9udC13ZWlnaHQ6IDcwMDsgbGV0dGVyLXNwYWNpbmc6IC0wLjVweDsgbWFyZ2luLWJvdHRvbTogNnB4OyB9CiAgICAudGFnbGluZSB7IGZvbnQtc2l6ZTogMTRweDsgY29sb3I6IHZhcigtLXRleHQtZGltKTsgdGV4dC1hbGlnbjogY2VudGVyOyBsaW5lLWhlaWdodDogMS41OyBtYXJnaW4tYm90dG9tOiAzMnB4OyB9CgogICAgLyog4pSA4pSAIENBUkRTIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwogICAgLmNhcmQgeyBiYWNrZ3JvdW5kOiB2YXIoLS1jYXJkKTsgYm9yZGVyOiAwLjVweCBzb2xpZCB2YXIoLS1ib3JkZXIpOyBib3JkZXItcmFkaXVzOiAxOHB4OyBwYWRkaW5nOiAxOHB4OyB3aWR0aDogMTAwJTsgbWFyZ2luLWJvdHRvbTogMTRweDsgfQogICAgLmNhcmQtbGFiZWwgeyBmb250LXNpemU6IDEwcHg7IGZvbnQtd2VpZ2h0OiA3MDA7IGNvbG9yOiB2YXIoLS10ZXh0LWZhaW50KTsgdGV4dC10cmFuc2Zvcm06IHVwcGVyY2FzZTsgbGV0dGVyLXNwYWNpbmc6IDAuNXB4OyBtYXJnaW4tYm90dG9tOiAxMHB4OyB9CiAgICAuY2FyZC12YWx1ZSB7IGZvbnQtc2l6ZTogMTVweDsgZm9udC13ZWlnaHQ6IDYwMDsgY29sb3I6IHZhcigtLXRleHQpOyB9CgogICAgLyog4pSA4pSAIExBTkcgUElMTFMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCiAgICAubGFuZy1zY3JvbGwgeyBkaXNwbGF5OiBmbGV4OyBnYXA6IDhweDsgb3ZlcmZsb3cteDogYXV0bzsgcGFkZGluZy1ib3R0b206IDRweDsgc2Nyb2xsYmFyLXdpZHRoOiBub25lOyB9CiAgICAubGFuZy1zY3JvbGw6Oi13ZWJraXQtc2Nyb2xsYmFyIHsgZGlzcGxheTogbm9uZTsgfQogICAgLmxhbmctcGlsbCB7IGZsZXgtc2hyaW5rOiAwOyBwYWRkaW5nOiA3cHggMTRweDsgYm9yZGVyLXJhZGl1czogMjBweDsgYm9yZGVyOiAwLjVweCBzb2xpZCB2YXIoLS1ib3JkZXIpOyBiYWNrZ3JvdW5kOiByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyBmb250LXNpemU6IDEzcHg7IGNvbG9yOiB2YXIoLS10ZXh0LWRpbSk7IGN1cnNvcjogcG9pbnRlcjsgd2hpdGUtc3BhY2U6IG5vd3JhcDsgdHJhbnNpdGlvbjogYWxsIDAuMTVzOyB9CiAgICAubGFuZy1waWxsLmFjdGl2ZS1hIHsgYm9yZGVyLWNvbG9yOiB2YXIoLS1wdXJwbGUpOyBiYWNrZ3JvdW5kOiByZ2JhKDEyOSwxNDAsMjQ4LDAuMTUpOyBjb2xvcjogdmFyKC0tcHVycGxlKTsgfQogICAgLmxhbmctcGlsbC5hY3RpdmUtYiB7IGJvcmRlci1jb2xvcjogdmFyKC0tcHVycGxlLWxpZ2h0KTsgYmFja2dyb3VuZDogcmdiYSgxOTYsMTgxLDI1MywwLjE1KTsgY29sb3I6IHZhcigtLXB1cnBsZS1saWdodCk7IH0KICAgIC5sYW5nLXBpbGw6aG92ZXIgeyBvcGFjaXR5OiAwLjg7IH0KCiAgICAvKiDilIDilIAgQlVUVE9OUyDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAgKi8KICAgIC5idG4geyB3aWR0aDogMTAwJTsgcGFkZGluZzogMTVweDsgYm9yZGVyLXJhZGl1czogMTRweDsgYm9yZGVyOiBub25lOyBmb250LXNpemU6IDE1cHg7IGZvbnQtd2VpZ2h0OiA3MDA7IGN1cnNvcjogcG9pbnRlcjsgdHJhbnNpdGlvbjogb3BhY2l0eSAwLjE1czsgfQogICAgLmJ0bjpob3ZlciB7IG9wYWNpdHk6IDAuODU7IH0KICAgIC5idG4tcHJpbWFyeSB7IGJhY2tncm91bmQ6IHZhcigtLXB1cnBsZSk7IGNvbG9yOiAjZmZmOyB9CiAgICAuYnRuLXNlY29uZGFyeSB7IGJhY2tncm91bmQ6IHJnYmEoMjU1LDI1NSwyNTUsMC4wNik7IGJvcmRlcjogMC41cHggc29saWQgdmFyKC0tYm9yZGVyKTsgY29sb3I6IHZhcigtLXRleHQpOyB9CiAgICAuYnRuLWRhbmdlciB7IGJhY2tncm91bmQ6IHJnYmEoMjM5LDY4LDY4LDAuMTIpOyBib3JkZXI6IDAuNXB4IHNvbGlkIHJnYmEoMjM5LDY4LDY4LDAuMyk7IGNvbG9yOiB2YXIoLS1yZWQpOyB9CiAgICAuYnRuLWdyZWVuIHsgYmFja2dyb3VuZDogcmdiYSg1MiwxOTksODksMC4xMik7IGJvcmRlcjogMC41cHggc29saWQgcmdiYSg1MiwxOTksODksMC4zKTsgY29sb3I6IHZhcigtLWdyZWVuKTsgfQogICAgLmJ0bjpkaXNhYmxlZCB7IG9wYWNpdHk6IDAuMzU7IGN1cnNvcjogbm90LWFsbG93ZWQ7IH0KCiAgICAvKiDilIDilIAgU1RBVFVTIEJBREdFIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwogICAgLnN0YXR1cy1iYWRnZSB7IGRpc3BsYXk6IGZsZXg7IGFsaWduLWl0ZW1zOiBjZW50ZXI7IGdhcDogN3B4OyBwYWRkaW5nOiA1cHggMTRweDsgYm9yZGVyLXJhZGl1czogMjBweDsgYm9yZGVyOiAwLjVweCBzb2xpZCB2YXIoLS1ib3JkZXIpOyBmb250LXNpemU6IDEycHg7IGZvbnQtd2VpZ2h0OiA1MDA7IGNvbG9yOiB2YXIoLS10ZXh0LWRpbSk7IH0KICAgIC5zdGF0dXMtYmFkZ2UuYWN0aXZlIHsgYm9yZGVyLWNvbG9yOiByZ2JhKDUyLDE5OSw4OSwwLjMpOyBiYWNrZ3JvdW5kOiByZ2JhKDUyLDE5OSw4OSwwLjA4KTsgY29sb3I6IHZhcigtLWdyZWVuKTsgfQogICAgLnN0YXR1cy1iYWRnZS5jb25uZWN0aW5nIHsgYm9yZGVyLWNvbG9yOiByZ2JhKDI1MSwxOTEsMzYsMC4zKTsgYmFja2dyb3VuZDogcmdiYSgyNTEsMTkxLDM2LDAuMDgpOyBjb2xvcjogI0ZCYmYyNDsgfQogICAgLnN0YXR1cy1kb3QgeyB3aWR0aDogNnB4OyBoZWlnaHQ6IDZweDsgYm9yZGVyLXJhZGl1czogNTAlOyBiYWNrZ3JvdW5kOiB2YXIoLS10ZXh0LWZhaW50KTsgfQogICAgLnN0YXR1cy1iYWRnZS5hY3RpdmUgLnN0YXR1cy1kb3QgeyBiYWNrZ3JvdW5kOiB2YXIoLS1ncmVlbik7IH0KICAgIC5zdGF0dXMtYmFkZ2UuY29ubmVjdGluZyAuc3RhdHVzLWRvdCB7IGJhY2tncm91bmQ6ICNGQmJmMjQ7IH0KCiAgICAvKiDilIDilIAgSEVBREVSIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwogICAgLnNlc3Npb24taGVhZGVyIHsgZGlzcGxheTogZmxleDsgYWxpZ24taXRlbXM6IGNlbnRlcjsganVzdGlmeS1jb250ZW50OiBzcGFjZS1iZXR3ZWVuOyB3aWR0aDogMTAwJTsgcGFkZGluZzogMTJweCAwIDhweDsgfQoKICAgIC8qIOKUgOKUgCBDSEFUIEJVQkJMRVMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCiAgICAuY2hhdC1ib3ggeyBmbGV4OiAxOyB3aWR0aDogMTAwJTsgb3ZlcmZsb3cteTogYXV0bzsgcGFkZGluZzogOHB4IDA7IG1pbi1oZWlnaHQ6IDIwMHB4OyB9CiAgICAuYnViYmxlLXJvdyB7IGRpc3BsYXk6IGZsZXg7IGFsaWduLWl0ZW1zOiBmbGV4LWVuZDsgZ2FwOiA4cHg7IG1hcmdpbi1ib3R0b206IDEycHg7IH0KICAgIC5idWJibGUtcm93LmZyb20taG9zdCB7IGZsZXgtZGlyZWN0aW9uOiByb3c7IH0KICAgIC5idWJibGUtcm93LmZyb20tZ3Vlc3QgeyBmbGV4LWRpcmVjdGlvbjogcm93LXJldmVyc2U7IH0KICAgIC5idWJibGUtYXZhdGFyIHsgd2lkdGg6IDMwcHg7IGhlaWdodDogMzBweDsgYm9yZGVyLXJhZGl1czogNTAlOyBkaXNwbGF5OiBmbGV4OyBhbGlnbi1pdGVtczogY2VudGVyOyBqdXN0aWZ5LWNvbnRlbnQ6IGNlbnRlcjsgZm9udC1zaXplOiAxMnB4OyBmb250LXdlaWdodDogNzAwOyBmbGV4LXNocmluazogMDsgbWFyZ2luLWJvdHRvbTogMnB4OyB9CiAgICAuYnViYmxlIHsgbWF4LXdpZHRoOiA3NSU7IGJvcmRlci1yYWRpdXM6IDE4cHg7IHBhZGRpbmc6IDExcHggMTRweDsgfQogICAgLmJ1YmJsZS1yb3cuZnJvbS1ob3N0IC5idWJibGUgeyBiYWNrZ3JvdW5kOiByZ2JhKDEyOSwxNDAsMjQ4LDAuMTUpOyBib3JkZXItYm90dG9tLWxlZnQtcmFkaXVzOiA0cHg7IH0KICAgIC5idWJibGUtcm93LmZyb20tZ3Vlc3QgLmJ1YmJsZSB7IGJhY2tncm91bmQ6IHJnYmEoMTk2LDE4MSwyNTMsMC4xNSk7IGJvcmRlci1ib3R0b20tcmlnaHQtcmFkaXVzOiA0cHg7IH0KICAgIC5idWJibGUtbmFtZSB7IGZvbnQtc2l6ZTogMTBweDsgZm9udC13ZWlnaHQ6IDcwMDsgbWFyZ2luLWJvdHRvbTogNHB4OyB9CiAgICAuYnViYmxlLXJvdy5mcm9tLWhvc3QgLmJ1YmJsZS1uYW1lIHsgY29sb3I6IHZhcigtLXB1cnBsZSk7IH0KICAgIC5idWJibGUtcm93LmZyb20tZ3Vlc3QgLmJ1YmJsZS1uYW1lIHsgY29sb3I6IHZhcigtLXB1cnBsZS1saWdodCk7IH0KICAgIC5idWJibGUtb3JpZ2luYWwgeyBmb250LXNpemU6IDEzcHg7IGNvbG9yOiByZ2JhKDI1NSwyNTUsMjU1LDAuOCk7IGxpbmUtaGVpZ2h0OiAxLjQ7IH0KICAgIC5idWJibGUtZGl2aWRlciB7IGhlaWdodDogMC41cHg7IGJhY2tncm91bmQ6IHZhcigtLWJvcmRlcik7IG1hcmdpbjogNnB4IDA7IH0KICAgIC5idWJibGUtdHJhbnNsYXRlZCB7IGZvbnQtc2l6ZTogMTJweDsgY29sb3I6IHJnYmEoMjU1LDI1NSwyNTUsMC41KTsgZm9udC1zdHlsZTogaXRhbGljOyBsaW5lLWhlaWdodDogMS40OyB9CiAgICAuYnViYmxlLXJlcGVhdCB7IGJhY2tncm91bmQ6IG5vbmU7IGJvcmRlcjogbm9uZTsgY3Vyc29yOiBwb2ludGVyOyBmbG9hdDogcmlnaHQ7IG1hcmdpbi10b3A6IDRweDsgZm9udC1zaXplOiAxMnB4OyBvcGFjaXR5OiAwLjY7IH0KICAgIC5idWJibGUtcmVwZWF0OmhvdmVyIHsgb3BhY2l0eTogMTsgfQoKICAgIC8qIOKUgOKUgCBSRUNPUkRJTkcgSU5ESUNBVE9SIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwogICAgLnJlY29yZGluZy1iYXIgeyBkaXNwbGF5OiBmbGV4OyBhbGlnbi1pdGVtczogY2VudGVyOyBnYXA6IDEwcHg7IGJhY2tncm91bmQ6IHZhcigtLWNhcmQpOyBib3JkZXItcmFkaXVzOiAxMnB4OyBwYWRkaW5nOiAxMHB4IDE0cHg7IHdpZHRoOiAxMDAlOyBtYXJnaW4tYm90dG9tOiA4cHg7IH0KICAgIC5yZWNvcmRpbmctZG90IHsgd2lkdGg6IDEwcHg7IGhlaWdodDogMTBweDsgYm9yZGVyLXJhZGl1czogNTAlOyBmbGV4LXNocmluazogMDsgfQogICAgLnJlY29yZGluZy10ZXh0IHsgZm9udC1zaXplOiAxMXB4OyBjb2xvcjogdmFyKC0tdGV4dC1kaW0pOyBmbGV4OiAxOyB9CiAgICAuYXVkaW8tbGV2ZWwgeyBoZWlnaHQ6IDNweDsgYmFja2dyb3VuZDogcmdiYSgyNTUsMjU1LDI1NSwwLjA4KTsgYm9yZGVyLXJhZGl1czogMnB4OyBvdmVyZmxvdzogaGlkZGVuOyBtYXJnaW4tdG9wOiA0cHg7IH0KICAgIC5hdWRpby1sZXZlbC1maWxsIHsgaGVpZ2h0OiAzcHg7IGJhY2tncm91bmQ6IHZhcigtLXB1cnBsZSk7IGJvcmRlci1yYWRpdXM6IDJweDsgdHJhbnNpdGlvbjogd2lkdGggMC4xczsgd2lkdGg6IDAlOyB9CgogICAgLyog4pSA4pSAIFNQRUVEIFNFTEVDVE9SIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwogICAgLnNwZWVkLXJvdyB7IGRpc3BsYXk6IGZsZXg7IGFsaWduLWl0ZW1zOiBjZW50ZXI7IGdhcDogNnB4OyB3aWR0aDogMTAwJTsgbWFyZ2luLWJvdHRvbTogOHB4OyB9CiAgICAuc3BlZWQtbGFiZWwgeyBmb250LXNpemU6IDExcHg7IGNvbG9yOiB2YXIoLS10ZXh0LWZhaW50KTsgbWFyZ2luLXJpZ2h0OiA0cHg7IH0KICAgIC5zcGVlZC1idG4geyBwYWRkaW5nOiA1cHggMTBweDsgYm9yZGVyLXJhZGl1czogMTJweDsgYm9yZGVyOiAwLjVweCBzb2xpZCB2YXIoLS1ib3JkZXIpOyBiYWNrZ3JvdW5kOiByZ2JhKDI1NSwyNTUsMjU1LDAuMDQpOyBmb250LXNpemU6IDExcHg7IGNvbG9yOiB2YXIoLS10ZXh0LWRpbSk7IGN1cnNvcjogcG9pbnRlcjsgZm9udC13ZWlnaHQ6IDYwMDsgfQogICAgLnNwZWVkLWJ0bi5hY3RpdmUgeyBib3JkZXItY29sb3I6IHZhcigtLXB1cnBsZSk7IGJhY2tncm91bmQ6IHJnYmEoMTI5LDE0MCwyNDgsMC4xNSk7IGNvbG9yOiB2YXIoLS1wdXJwbGUpOyB9CgogICAgLyog4pSA4pSAIFdBUk5JTkcg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCiAgICAud2FybmluZy1ib3ggeyBiYWNrZ3JvdW5kOiByZ2JhKDIzOSw2OCw2OCwwLjA4KTsgYm9yZGVyOiAwLjVweCBzb2xpZCByZ2JhKDIzOSw2OCw2OCwwLjMpOyBib3JkZXItcmFkaXVzOiAxNHB4OyBwYWRkaW5nOiAxNHB4OyB3aWR0aDogMTAwJTsgbWFyZ2luLWJvdHRvbTogMTRweDsgZm9udC1zaXplOiAxM3B4OyBjb2xvcjogcmdiYSgyNTUsMjU1LDI1NSwwLjUpOyBsaW5lLWhlaWdodDogMS41OyB9CiAgICAuaW5mby1ib3ggeyBiYWNrZ3JvdW5kOiByZ2JhKDEyOSwxNDAsMjQ4LDAuMDgpOyBib3JkZXI6IDAuNXB4IHNvbGlkIHJnYmEoMTI5LDE0MCwyNDgsMC4yNSk7IGJvcmRlci1yYWRpdXM6IDE0cHg7IHBhZGRpbmc6IDE0cHg7IHdpZHRoOiAxMDAlOyBtYXJnaW4tYm90dG9tOiAxNHB4OyBmb250LXNpemU6IDEzcHg7IGNvbG9yOiByZ2JhKDI1NSwyNTUsMjU1LDAuNik7IGxpbmUtaGVpZ2h0OiAxLjU7IH0KCiAgICAvKiDilIDilIAgRU1QVFkgU1RBVEUg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCiAgICAuZW1wdHktdGV4dCB7IGZvbnQtc2l6ZTogMTJweDsgY29sb3I6IHZhcigtLXRleHQtZmFpbnQpOyB0ZXh0LWFsaWduOiBjZW50ZXI7IHBhZGRpbmc6IDMwcHggMDsgbGluZS1oZWlnaHQ6IDEuNzsgfQoKICAgIC8qIOKUgOKUgCBISU5UIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwogICAgLmhpbnQgeyBmb250LXNpemU6IDExcHg7IGNvbG9yOiB2YXIoLS10ZXh0LWZhaW50KTsgdGV4dC1hbGlnbjogY2VudGVyOyBwYWRkaW5nOiAxMHB4IDIwcHg7IH0KCiAgICAvKiDilIDilIAgTEFORyBJTkZPIFJPVyDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAgKi8KICAgIC5sYW5nLWluZm8tcm93IHsgZGlzcGxheTogZmxleDsgYWxpZ24taXRlbXM6IGNlbnRlcjsgZ2FwOiA4cHg7IHdpZHRoOiAxMDAlOyBtYXJnaW4tYm90dG9tOiAxNHB4OyB9CiAgICAubGFuZy1pbmZvLWNhcmQgeyBmbGV4OiAxOyBiYWNrZ3JvdW5kOiB2YXIoLS1jYXJkKTsgYm9yZGVyOiAxLjVweCBzb2xpZCB2YXIoLS1ib3JkZXIpOyBib3JkZXItcmFkaXVzOiAxNnB4OyBwYWRkaW5nOiAxMnB4OyB0ZXh0LWFsaWduOiBjZW50ZXI7IH0KICAgIC5sYW5nLWluZm8taWNvbiB7IGZvbnQtc2l6ZTogMjBweDsgbWFyZ2luLWJvdHRvbTogNHB4OyB9CiAgICAubGFuZy1pbmZvLWxhYmVsIHsgZm9udC1zaXplOiAxMHB4OyBmb250LXdlaWdodDogNzAwOyB0ZXh0LXRyYW5zZm9ybTogdXBwZXJjYXNlOyBsZXR0ZXItc3BhY2luZzogMC41cHg7IG1hcmdpbi1ib3R0b206IDRweDsgfQogICAgLmxhbmctaW5mby1uYW1lIHsgZm9udC1zaXplOiAxMnB4OyBjb2xvcjogcmdiYSgyNTUsMjU1LDI1NSwwLjYpOyB9CiAgICAubGFuZy1hcnJvdyB7IGZvbnQtc2l6ZTogMThweDsgY29sb3I6IHZhcigtLXRleHQtZmFpbnQpOyB9CgogICAgLyog4pSA4pSAIFNQSU5ORVIg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCiAgICAuc3Bpbm5lciB7IHdpZHRoOiA0MHB4OyBoZWlnaHQ6IDQwcHg7IGJvcmRlcjogM3B4IHNvbGlkIHZhcigtLWJvcmRlcik7IGJvcmRlci10b3AtY29sb3I6IHZhcigtLXB1cnBsZSk7IGJvcmRlci1yYWRpdXM6IDUwJTsgYW5pbWF0aW9uOiBzcGluIDAuOHMgbGluZWFyIGluZmluaXRlOyBtYXJnaW46IDIwcHggYXV0bzsgfQogICAgQGtleWZyYW1lcyBzcGluIHsgdG8geyB0cmFuc2Zvcm06IHJvdGF0ZSgzNjBkZWcpOyB9IH0KCiAgICAvKiDilIDilIAgRk9PVEVSIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwogICAgLmZvb3RlciB7IGZvbnQtc2l6ZTogMTFweDsgY29sb3I6IHZhcigtLXRleHQtZmFpbnQpOyB0ZXh0LWFsaWduOiBjZW50ZXI7IHBhZGRpbmc6IDIwcHggMCAxMnB4OyB9CiAgICAuZm9vdGVyIGEgeyBjb2xvcjogcmdiYSg5OSwxMDIsMjQxLDAuNyk7IHRleHQtZGVjb3JhdGlvbjogbm9uZTsgfQoKICAgIC8qIOKUgOKUgCBESVZJREVSIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwogICAgLnNlY3Rpb24tbGFiZWwgeyBmb250LXNpemU6IDEwcHg7IGZvbnQtd2VpZ2h0OiA2MDA7IGNvbG9yOiB2YXIoLS10ZXh0LWZhaW50KTsgdGV4dC10cmFuc2Zvcm06IHVwcGVyY2FzZTsgbGV0dGVyLXNwYWNpbmc6IDAuNXB4OyB3aWR0aDogMTAwJTsgbWFyZ2luLWJvdHRvbTogNnB4OyBtYXJnaW4tdG9wOiA0cHg7IH0KCiAgICAvKiDilIDilIAgTk9USUZJQ0FUSU9OIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwogICAgLm5vdGlmIHsgcG9zaXRpb246IGZpeGVkOyB0b3A6IDE2cHg7IGxlZnQ6IDUwJTsgdHJhbnNmb3JtOiB0cmFuc2xhdGVYKC01MCUpOyBiYWNrZ3JvdW5kOiAjMUExQTJFOyBib3JkZXI6IDAuNXB4IHNvbGlkIHZhcigtLXB1cnBsZSk7IGJvcmRlci1yYWRpdXM6IDEycHg7IHBhZGRpbmc6IDEycHggMjBweDsgZm9udC1zaXplOiAxM3B4OyBjb2xvcjogdmFyKC0tdGV4dCk7IHotaW5kZXg6IDk5OTsgbWF4LXdpZHRoOiAzNDBweDsgdGV4dC1hbGlnbjogY2VudGVyOyBib3gtc2hhZG93OiAwIDhweCAyNHB4IHJnYmEoMCwwLDAsMC40KTsgZGlzcGxheTogbm9uZTsgfQogICAgLm5vdGlmLnNob3cgeyBkaXNwbGF5OiBibG9jazsgfQogIDwvc3R5bGU+CjwvaGVhZD4KPGJvZHk+Cgo8IS0tIOKUgOKUgCBTQ1JFRU46IExPQURJTkcg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAIC0tPgo8ZGl2IGlkPSJzY3JlZW4tbG9hZGluZyIgY2xhc3M9InNjcmVlbiBhY3RpdmUiPgogIDxkaXYgY2xhc3M9ImxvZ28td3JhcCI+8J+MkDwvZGl2PgogIDxkaXYgY2xhc3M9ImFwcC1uYW1lIj5QYXJsb3JhIEFJPC9kaXY+CiAgPGRpdiBjbGFzcz0ic3Bpbm5lciI+PC9kaXY+CiAgPGRpdiBjbGFzcz0idGFnbGluZSIgaWQ9ImxvYWRpbmctdGV4dCI+Q29ubmVjdGluZyB0byByb29tLi4uPC9kaXY+CjwvZGl2PgoKPCEtLSDilIDilIAgU0NSRUVOOiBDT05GSVJNIExBTkdTIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAtLT4KPGRpdiBpZD0ic2NyZWVuLWNvbmZpcm0iIGNsYXNzPSJzY3JlZW4iPgogIDxkaXYgY2xhc3M9ImxvZ28td3JhcCI+8J+MkDwvZGl2PgogIDxkaXYgY2xhc3M9ImFwcC1uYW1lIj5QYXJsb3JhIEFJPC9kaXY+CiAgPGRpdiBjbGFzcz0idGFnbGluZSI+UmVhbC10aW1lIHNpbXVsdGFuZW91cyB0cmFuc2xhdGlvbjwvZGl2PgoKICA8ZGl2IGNsYXNzPSJpbmZvLWJveCIgaWQ9ImFuYy13YXJuaW5nIj4KICAgIPCfjqcgPHN0cm9uZz5Gb3IgYmVzdCBxdWFsaXR5PC9zdHJvbmc+IHVzZSBub2lzZS1jYW5jZWxsaW5nIGVhcnBob25lcyAoQU5DKS4gV2l0aG91dCB0aGVtLCB5b3VyIG1pYyBtYXkgcGljayB1cCB0aGUgdHJhbnNsYXRpb24gYXVkaW8uCiAgPC9kaXY+CgogIDxkaXYgY2xhc3M9ImNhcmQiPgogICAgPGRpdiBjbGFzcz0iY2FyZC1sYWJlbCI+WW91IHdpbGwgc3BlYWsgaW48L2Rpdj4KICAgIDxkaXYgY2xhc3M9Imxhbmctc2Nyb2xsIiBpZD0iZ3Vlc3Qtc3BlYWstbGFuZ3MiPjwvZGl2PgogIDwvZGl2PgoKICA8ZGl2IGNsYXNzPSJjYXJkIj4KICAgIDxkaXYgY2xhc3M9ImNhcmQtbGFiZWwiPllvdSB3aWxsIGhlYXIgdGhlIHRyYW5zbGF0aW9uIGluPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJsYW5nLXNjcm9sbCIgaWQ9Imd1ZXN0LWhlYXItbGFuZ3MiPjwvZGl2PgogIDwvZGl2PgoKICA8ZGl2IGNsYXNzPSJ3YXJuaW5nLWJveCIgaWQ9Imxhbmctd2FybmluZyIgc3R5bGU9ImRpc3BsYXk6bm9uZSI+CiAgICDimqDvuI8gWW91IHNlbGVjdGVkIHRoZSBzYW1lIGxhbmd1YWdlIGZvciBzcGVha2luZyBhbmQgaGVhcmluZy4gUGxlYXNlIGNob29zZSBkaWZmZXJlbnQgbGFuZ3VhZ2VzLgogIDwvZGl2PgoKICA8YnV0dG9uIGNsYXNzPSJidG4gYnRuLXByaW1hcnkiIGlkPSJidG4tam9pbiIgc3R5bGU9Im1hcmdpbi10b3A6OHB4Ij5Kb2luIHNlc3Npb24g4oaSPC9idXR0b24+CgogIDxkaXYgY2xhc3M9ImZvb3RlciI+CiAgICBQb3dlcmVkIGJ5IDxhIGhyZWY9Imh0dHBzOi8vcGFybG9yYS5haSIgdGFyZ2V0PSJfYmxhbmsiPlBhcmxvcmEgQUk8L2E+CiAgPC9kaXY+CjwvZGl2PgoKPCEtLSDilIDilIAgU0NSRUVOOiBTRVNTSU9OIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAtLT4KPGRpdiBpZD0ic2NyZWVuLXNlc3Npb24iIGNsYXNzPSJzY3JlZW4iPgogIDxkaXYgY2xhc3M9InNlc3Npb24taGVhZGVyIj4KICAgIDxidXR0b24gY2xhc3M9ImJ0biBidG4tc2Vjb25kYXJ5IiBpZD0iYnRuLWxlYXZlIiBzdHlsZT0id2lkdGg6YXV0bztwYWRkaW5nOjhweCAxNnB4O2ZvbnQtc2l6ZToxM3B4Ij5MZWF2ZTwvYnV0dG9uPgogICAgPGRpdiBjbGFzcz0ic3RhdHVzLWJhZGdlIGNvbm5lY3RpbmciIGlkPSJzdGF0dXMtYmFkZ2UiPgogICAgICA8ZGl2IGNsYXNzPSJzdGF0dXMtZG90Ij48L2Rpdj4KICAgICAgPHNwYW4gaWQ9InN0YXR1cy10ZXh0Ij5Db25uZWN0aW5nLi4uPC9zcGFuPgogICAgPC9kaXY+CiAgICA8YnV0dG9uIGNsYXNzPSJidG4gYnRuLXNlY29uZGFyeSIgaWQ9ImJ0bi1yZXBlYXQiIHN0eWxlPSJ3aWR0aDphdXRvO3BhZGRpbmc6OHB4IDEycHg7Zm9udC1zaXplOjE2cHg7b3BhY2l0eTowLjQiIGRpc2FibGVkPvCflIE8L2J1dHRvbj4KICA8L2Rpdj4KCiAgPGRpdiBjbGFzcz0ibGFuZy1pbmZvLXJvdyIgaWQ9ImxhbmctaW5mby1yb3ciPgogICAgPCEtLSBmaWxsZWQgYnkgSlMgLS0+CiAgPC9kaXY+CgogIDxkaXYgY2xhc3M9InJlY29yZGluZy1iYXIiIGlkPSJyZWNvcmRpbmctYmFyIiBzdHlsZT0iZGlzcGxheTpub25lIj4KICAgIDxkaXYgY2xhc3M9InJlY29yZGluZy1kb3QiIGlkPSJyZWNvcmRpbmctZG90IiBzdHlsZT0iYmFja2dyb3VuZDojRUY0NDQ0Ij48L2Rpdj4KICAgIDxkaXYgc3R5bGU9ImZsZXg6MSI+CiAgICAgIDxkaXYgY2xhc3M9InJlY29yZGluZy10ZXh0IiBpZD0icmVjb3JkaW5nLXRleHQiPkNhcHR1cmluZyBhdWRpby4uLjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJhdWRpby1sZXZlbCI+PGRpdiBjbGFzcz0iYXVkaW8tbGV2ZWwtZmlsbCIgaWQ9ImF1ZGlvLWxldmVsLWZpbGwiPjwvZGl2PjwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+CgogIDxkaXYgY2xhc3M9InNwZWVkLXJvdyI+CiAgICA8c3BhbiBjbGFzcz0ic3BlZWQtbGFiZWwiPlNwZWVkOjwvc3Bhbj4KICAgIDxidXR0b24gY2xhc3M9InNwZWVkLWJ0biIgZGF0YS1zcGVlZD0iMC41Ij4wLjV4PC9idXR0b24+CiAgICA8YnV0dG9uIGNsYXNzPSJzcGVlZC1idG4iIGRhdGEtc3BlZWQ9IjAuNzUiPjAuNzV4PC9idXR0b24+CiAgICA8YnV0dG9uIGNsYXNzPSJzcGVlZC1idG4gYWN0aXZlIiBkYXRhLXNwZWVkPSIxIj4xeDwvYnV0dG9uPgogICAgPGJ1dHRvbiBjbGFzcz0ic3BlZWQtYnRuIiBkYXRhLXNwZWVkPSIxLjI1Ij4xLjI1eDwvYnV0dG9uPgogICAgPGJ1dHRvbiBjbGFzcz0ic3BlZWQtYnRuIiBkYXRhLXNwZWVkPSIxLjUiPjEuNXg8L2J1dHRvbj4KICA8L2Rpdj4KCiAgPGRpdiBjbGFzcz0ic2VjdGlvbi1sYWJlbCI+TGl2ZSB0cmFuc2xhdGlvbjwvZGl2PgogIDxkaXYgY2xhc3M9ImNoYXQtYm94IiBpZD0iY2hhdC1ib3giPgogICAgPGRpdiBjbGFzcz0iZW1wdHktdGV4dCIgaWQ9ImVtcHR5LXRleHQiPgogICAgICBXYWl0aW5nIGZvciB0aGUgc2Vzc2lvbiB0byBzdGFydC4uLjxicj5NYWtlIHN1cmUgYm90aCBwYXJ0aWNpcGFudHMgYXJlIGNvbm5lY3RlZC4KICAgIDwvZGl2PgogIDwvZGl2PgoKICA8ZGl2IGNsYXNzPSJoaW50IiBpZD0ic2Vzc2lvbi1oaW50Ij7wn46nIEtlZXAgZWFycGhvbmVzIG9uIOKAlCB0cmFuc2xhdGlvbiB3aWxsIHBsYXkgYXV0b21hdGljYWxseTwvZGl2PgoKICA8YnV0dG9uIGNsYXNzPSJidG4gYnRuLWRhbmdlciIgaWQ9ImJ0bi1zdG9wIiBzdHlsZT0ibWFyZ2luLXRvcDo4cHg7ZGlzcGxheTpub25lIj7ij7kgU3RvcCBzZXNzaW9uPC9idXR0b24+CiAgPGJ1dHRvbiBjbGFzcz0iYnRuIGJ0bi1ncmVlbiIgaWQ9ImJ0bi1zdGFydCIgc3R5bGU9Im1hcmdpbi10b3A6OHB4Ij7ilrYgU3RhcnQ8L2J1dHRvbj4KPC9kaXY+Cgo8IS0tIOKUgOKUgCBTQ1JFRU46IEVOREVEIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAtLT4KPGRpdiBpZD0ic2NyZWVuLWVuZGVkIiBjbGFzcz0ic2NyZWVuIj4KICA8ZGl2IHN0eWxlPSJmbGV4OjE7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjthbGlnbi1pdGVtczpjZW50ZXI7anVzdGlmeS1jb250ZW50OmNlbnRlcjtnYXA6MTZweDt0ZXh0LWFsaWduOmNlbnRlciI+CiAgICA8ZGl2IHN0eWxlPSJmb250LXNpemU6NTZweCI+8J+RizwvZGl2PgogICAgPGRpdiBzdHlsZT0iZm9udC1zaXplOjIycHg7Zm9udC13ZWlnaHQ6NzAwIj5TZXNzaW9uIGVuZGVkPC9kaXY+CiAgICA8ZGl2IHN0eWxlPSJmb250LXNpemU6MTRweDtjb2xvcjp2YXIoLS10ZXh0LWRpbSkiIGlkPSJlbmRlZC1yZWFzb24iPlRoZSBzZXNzaW9uIGhhcyBiZWVuIGNsb3NlZC48L2Rpdj4KICA8L2Rpdj4KPC9kaXY+Cgo8IS0tIOKUgOKUgCBOT1RJRklDQVRJT04g4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAIC0tPgo8ZGl2IGNsYXNzPSJub3RpZiIgaWQ9Im5vdGlmIj48L2Rpdj4KCjxzY3JpcHQ+Ci8vIOKUgOKUgCBDT05GSUcg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmNvbnN0IEJBQ0tFTkRfSFRUUCA9ICcnOwpjb25zdCBCQUNLRU5EX1dTICAgPSBgd3NzOi8vJHtsb2NhdGlvbi5ob3N0fS93c2A7CmNvbnN0IENIVU5LX01TID0gNDAwMDsgLy8gNCBzZWd1bmRvcyBwb3IgY2h1bmsKCi8vIOKUgOKUgCBMQU5HVUFHRVMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmNvbnN0IExBTkdVQUdFUyA9IFsKICB7IGNvZGU6ICdFUycsIG5hbWU6ICdFc3Bhw7FvbCcsICAgIGZsYWc6ICfwn4eq8J+HuCcgfSwKICB7IGNvZGU6ICdFTicsIG5hbWU6ICdFbmdsaXNoJywgICAgZmxhZzogJ/Cfh6zwn4enJyB9LAogIHsgY29kZTogJ0ZSJywgbmFtZTogJ0ZyYW7Dp2FpcycsICAgZmxhZzogJ/Cfh6vwn4e3JyB9LAogIHsgY29kZTogJ0RFJywgbmFtZTogJ0RldXRzY2gnLCAgICBmbGFnOiAn8J+HqfCfh6onIH0sCiAgeyBjb2RlOiAnSVQnLCBuYW1lOiAnSXRhbGlhbm8nLCAgIGZsYWc6ICfwn4eu8J+HuScgfSwKICB7IGNvZGU6ICdQVCcsIG5hbWU6ICdQb3J0dWd1w6pzJywgIGZsYWc6ICfwn4e18J+HuScgfSwKICB7IGNvZGU6ICdTSycsIG5hbWU6ICdTbG92ZW7EjWluYScsIGZsYWc6ICfwn4e48J+HsCcgfSwKICB7IGNvZGU6ICdDUycsIG5hbWU6ICfEjGXFoXRpbmEnLCAgICBmbGFnOiAn8J+HqPCfh78nIH0sCiAgeyBjb2RlOiAnUEwnLCBuYW1lOiAnUG9sc2tpJywgICAgIGZsYWc6ICfwn4e18J+HsScgfSwKICB7IGNvZGU6ICdOTCcsIG5hbWU6ICdOZWRlcmxhbmRzJywgZmxhZzogJ/Cfh7Pwn4exJyB9LApdOwoKLy8g4pSA4pSAIFNUQVRFIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgApsZXQgcm9vbUlkID0gbnVsbDsKbGV0IHJvb21EYXRhID0gbnVsbDsKbGV0IGd1ZXN0U3BlYWtMYW5nID0gbnVsbDsgLy8gaWRpb21hIHF1ZSBoYWJsYSBlbCBndWVzdApsZXQgZ3Vlc3RIZWFyTGFuZyA9IG51bGw7ICAvLyBpZGlvbWEgcXVlIGVzY3VjaGEgZWwgZ3Vlc3QgKD0gbG8gcXVlIGhhYmxhIGVsIGhvc3QpCmxldCB3cyA9IG51bGw7CmxldCBpc0FjdGl2ZSA9IGZhbHNlOwpsZXQgbWVkaWFSZWNvcmRlciA9IG51bGw7CmxldCBhdWRpb0NvbnRleHQgPSBudWxsOwpsZXQgYW5hbHlzZXIgPSBudWxsOwpsZXQgYW5pbUZyYW1lSWQgPSBudWxsOwpsZXQgdHRzU3BlZWQgPSAxLjA7CmxldCBsYXN0VHJhbnNsYXRpb24gPSBudWxsOwpsZXQgY2h1bmtDb3VudCA9IDA7CmxldCBwaW5nSW50ZXJ2YWwgPSBudWxsOwoKLy8g4pSA4pSAIEhFTFBFUlMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmZ1bmN0aW9uIHNob3dTY3JlZW4oaWQpIHsKICBkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcuc2NyZWVuJykuZm9yRWFjaChzID0+IHMuY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJykpOwogIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzY3JlZW4tJyArIGlkKS5jbGFzc0xpc3QuYWRkKCdhY3RpdmUnKTsKfQoKZnVuY3Rpb24gc2hvd05vdGlmKG1zZywgZHVyYXRpb24gPSAzMDAwKSB7CiAgY29uc3QgZWwgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbm90aWYnKTsKICBlbC50ZXh0Q29udGVudCA9IG1zZzsKICBlbC5jbGFzc0xpc3QuYWRkKCdzaG93Jyk7CiAgc2V0VGltZW91dCgoKSA9PiBlbC5jbGFzc0xpc3QucmVtb3ZlKCdzaG93JyksIGR1cmF0aW9uKTsKfQoKZnVuY3Rpb24gZ2V0TGFuZyhjb2RlKSB7IHJldHVybiBMQU5HVUFHRVMuZmluZChsID0+IGwuY29kZSA9PT0gY29kZSkgfHwgeyBjb2RlLCBuYW1lOiBjb2RlLCBmbGFnOiAn8J+MkCcgfTsgfQoKZnVuY3Rpb24gc2V0U3RhdHVzKHRleHQsIHN0YXRlID0gJycpIHsKICBjb25zdCBiYWRnZSA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdzdGF0dXMtYmFkZ2UnKTsKICBjb25zdCBzcGFuID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3N0YXR1cy10ZXh0Jyk7CiAgYmFkZ2UuY2xhc3NOYW1lID0gJ3N0YXR1cy1iYWRnZScgKyAoc3RhdGUgPyAnICcgKyBzdGF0ZSA6ICcnKTsKICBzcGFuLnRleHRDb250ZW50ID0gdGV4dDsKfQoKZnVuY3Rpb24gYWRkQnViYmxlKGZyb20sIG9yaWdpbmFsLCB0cmFuc2xhdGVkLCBzcmNMYW5nLCB0Z3RMYW5nKSB7CiAgY29uc3QgYm94ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2NoYXQtYm94Jyk7CiAgY29uc3QgZW1wdHkgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZW1wdHktdGV4dCcpOwogIGlmIChlbXB0eSkgZW1wdHkucmVtb3ZlKCk7CgogIGNvbnN0IHNyY0wgPSBnZXRMYW5nKHNyY0xhbmcpOwogIGNvbnN0IHRndEwgPSBnZXRMYW5nKHRndExhbmcpOwogIGNvbnN0IGlzSG9zdCA9IGZyb20gPT09ICdob3N0JzsKICBjb25zdCBuYW1lID0gaXNIb3N0ID8gJ0hvc3QnIDogJ1lvdSc7CiAgY29uc3QgdHJhbnNsSWQgPSAndGwtJyArIERhdGUubm93KCk7CgogIGNvbnN0IHJvdyA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpOwogIHJvdy5jbGFzc05hbWUgPSAnYnViYmxlLXJvdyAnICsgKGlzSG9zdCA/ICdmcm9tLWhvc3QnIDogJ2Zyb20tZ3Vlc3QnKTsKICByb3cuaW5uZXJIVE1MID0gYAogICAgPGRpdiBjbGFzcz0iYnViYmxlLWF2YXRhciIgc3R5bGU9ImJhY2tncm91bmQ6JHtpc0hvc3QgPyAncmdiYSgxMjksMTQwLDI0OCwwLjIpJyA6ICdyZ2JhKDE5NiwxODEsMjUzLDAuMiknfSI+CiAgICAgICR7bmFtZS5jaGFyQXQoMCl9CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImJ1YmJsZSI+CiAgICAgIDxkaXYgY2xhc3M9ImJ1YmJsZS1uYW1lIj4ke25hbWV9PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImJ1YmJsZS1vcmlnaW5hbCI+JHtzcmNMLmZsYWd9ICR7ZXNjSHRtbChvcmlnaW5hbCl9PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImJ1YmJsZS1kaXZpZGVyIj48L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iYnViYmxlLXRyYW5zbGF0ZWQiIGlkPSIke3RyYW5zbElkfSI+JHt0Z3RMLmZsYWd9ICR7ZXNjSHRtbCh0cmFuc2xhdGVkKX08L2Rpdj4KICAgICAgPGJ1dHRvbiBjbGFzcz0iYnViYmxlLXJlcGVhdCIgb25jbGljaz0icmVwbGF5VHRzKCcke2VzY0h0bWwodHJhbnNsYXRlZCl9JywnJHt0Z3RMYW5nfScpIj7wn5SBPC9idXR0b24+CiAgICA8L2Rpdj4KICBgOwogIGJveC5hcHBlbmRDaGlsZChyb3cpOwogIGJveC5zY3JvbGxUb3AgPSBib3guc2Nyb2xsSGVpZ2h0OwoKICBsYXN0VHJhbnNsYXRpb24gPSB7IHRyYW5zbGF0ZWQsIHR0c0xvY2FsZTogdGd0TGFuZyB9OwogIGNvbnN0IHJlcGVhdEJ0biA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdidG4tcmVwZWF0Jyk7CiAgaWYgKHJlcGVhdEJ0bikgeyByZXBlYXRCdG4uZGlzYWJsZWQgPSBmYWxzZTsgcmVwZWF0QnRuLnN0eWxlLm9wYWNpdHkgPSAnMSc7IH0KfQoKZnVuY3Rpb24gZXNjSHRtbChzKSB7IHJldHVybiBzLnJlcGxhY2UoLyYvZywnJmFtcDsnKS5yZXBsYWNlKC88L2csJyZsdDsnKS5yZXBsYWNlKC8+L2csJyZndDsnKS5yZXBsYWNlKC8iL2csJyZxdW90OycpOyB9CgpmdW5jdGlvbiByZXBsYXlUdHModGV4dCwgbGFuZykgewogIGlmICghd2luZG93LnNwZWVjaFN5bnRoZXNpcykgcmV0dXJuOwogIHdpbmRvdy5zcGVlY2hTeW50aGVzaXMuY2FuY2VsKCk7CiAgY29uc3QgdXR0ID0gbmV3IFNwZWVjaFN5bnRoZXNpc1V0dGVyYW5jZSh0ZXh0KTsKICB1dHQubGFuZyA9IGxhbmdDb2RlVG9Mb2NhbGUobGFuZyk7CiAgdXR0LnJhdGUgPSB0dHNTcGVlZDsKICB3aW5kb3cuc3BlZWNoU3ludGhlc2lzLnNwZWFrKHV0dCk7Cn0KCmZ1bmN0aW9uIGxhbmdDb2RlVG9Mb2NhbGUoY29kZSkgewogIGNvbnN0IG1hcCA9IHsgRVM6J2VzLUVTJywgRU46J2VuLVVTJywgRlI6J2ZyLUZSJywgREU6J2RlLURFJywgSVQ6J2l0LUlUJywgUFQ6J3B0LVBUJywgU0s6J3NrLVNLJywgQ1M6J2NzLUNaJywgUEw6J3BsLVBMJywgTkw6J25sLU5MJyB9OwogIHJldHVybiBtYXBbY29kZV0gfHwgY29kZS50b0xvd2VyQ2FzZSgpOwp9CgovLyDilIDilIAgSU5JVCDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKYXN5bmMgZnVuY3Rpb24gaW5pdCgpIHsKICAvLyBMZWVyIHJvb21JZCBkZSBsYSBVUkw6ID9yb29tPVhYWFgKICBjb25zdCBwYXJhbXMgPSBuZXcgVVJMU2VhcmNoUGFyYW1zKHdpbmRvdy5sb2NhdGlvbi5zZWFyY2gpOwogIC8vIFRyeSA/cm9vbT1YWFhYWCBmaXJzdCwgdGhlbiBwYXRoIC9qb2luL1hYWFhYCiAgcm9vbUlkID0gcGFyYW1zLmdldCgncm9vbScpOwogIGlmICghcm9vbUlkKSB7CiAgICBjb25zdCBwYXRoUGFydHMgPSB3aW5kb3cubG9jYXRpb24ucGF0aG5hbWUuc3BsaXQoJy8nKTsKICAgIGNvbnN0IGxhc3RQYXJ0ID0gcGF0aFBhcnRzW3BhdGhQYXJ0cy5sZW5ndGggLSAxXTsKICAgIGlmIChsYXN0UGFydCAmJiBsYXN0UGFydC5sZW5ndGggPT09IDUgJiYgL15cZCskLy50ZXN0KGxhc3RQYXJ0KSkgewogICAgICByb29tSWQgPSBsYXN0UGFydDsKICAgIH0KICB9CgogIGlmICghcm9vbUlkKSB7CiAgICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbG9hZGluZy10ZXh0JykudGV4dENvbnRlbnQgPSAnSW52YWxpZCBsaW5rLiBQbGVhc2Ugc2NhbiB0aGUgUVIgY29kZSBhZ2Fpbi4nOwogICAgcmV0dXJuOwogIH0KCiAgLy8gRmV0Y2ggcm9vbSBpbmZvCiAgdHJ5IHsKICAgIGNvbnN0IHIgPSBhd2FpdCBmZXRjaChgJHtCQUNLRU5EX0hUVFB9L3Jvb20vJHtyb29tSWR9YCk7CiAgICBpZiAoIXIub2spIHRocm93IG5ldyBFcnJvcignUm9vbSBub3QgZm91bmQnKTsKICAgIHJvb21EYXRhID0gYXdhaXQgci5qc29uKCk7CiAgfSBjYXRjaCAoZSkgewogICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2xvYWRpbmctdGV4dCcpLnRleHRDb250ZW50ID0gJ1Jvb20gbm90IGZvdW5kIG9yIGV4cGlyZWQuIFBsZWFzZSBhc2sgdGhlIGhvc3QgdG8gY3JlYXRlIGEgbmV3IHNlc3Npb24uJzsKICAgIHJldHVybjsKICB9CgogIC8vIFBvciBkZWZlY3RvOiBlbCBndWVzdCBoYWJsYSBlbiBsYW5nR3Vlc3QgeSBlc2N1Y2hhIGxhbmdIb3N0CiAgZ3Vlc3RTcGVha0xhbmcgPSByb29tRGF0YS5sYW5nR3Vlc3Q7CiAgZ3Vlc3RIZWFyTGFuZyA9IHJvb21EYXRhLmxhbmdIb3N0OwoKICBidWlsZENvbmZpcm1TY3JlZW4oKTsKICBzaG93U2NyZWVuKCdjb25maXJtJyk7Cn0KCmZ1bmN0aW9uIGJ1aWxkQ29uZmlybVNjcmVlbigpIHsKICAvLyBTZWxlY3RvciBpZGlvbWEgaGFibGFyIChwb3IgZGVmZWN0byBsYW5nR3Vlc3QgZGVsIGhvc3QpCiAgY29uc3Qgc3BlYWtDb250YWluZXIgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnZ3Vlc3Qtc3BlYWstbGFuZ3MnKTsKICBjb25zdCBoZWFyQ29udGFpbmVyID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2d1ZXN0LWhlYXItbGFuZ3MnKTsKCiAgTEFOR1VBR0VTLmZvckVhY2gobCA9PiB7CiAgICBjb25zdCBwaWxsMSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2J1dHRvbicpOwogICAgcGlsbDEuY2xhc3NOYW1lID0gJ2xhbmctcGlsbCcgKyAobC5jb2RlID09PSBndWVzdFNwZWFrTGFuZyA/ICcgYWN0aXZlLWInIDogJycpOwogICAgcGlsbDEudGV4dENvbnRlbnQgPSBsLmZsYWcgKyAnICcgKyBsLm5hbWU7CiAgICBwaWxsMS5vbmNsaWNrID0gKCkgPT4gewogICAgICBndWVzdFNwZWFrTGFuZyA9IGwuY29kZTsKICAgICAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI2d1ZXN0LXNwZWFrLWxhbmdzIC5sYW5nLXBpbGwnKS5mb3JFYWNoKHAgPT4gcC5jbGFzc05hbWUgPSAnbGFuZy1waWxsJyk7CiAgICAgIHBpbGwxLmNsYXNzTmFtZSA9ICdsYW5nLXBpbGwgYWN0aXZlLWInOwogICAgICB2YWxpZGF0ZUxhbmdzKCk7CiAgICB9OwogICAgc3BlYWtDb250YWluZXIuYXBwZW5kQ2hpbGQocGlsbDEpOwoKICAgIGNvbnN0IHBpbGwyID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnYnV0dG9uJyk7CiAgICBwaWxsMi5jbGFzc05hbWUgPSAnbGFuZy1waWxsJyArIChsLmNvZGUgPT09IGd1ZXN0SGVhckxhbmcgPyAnIGFjdGl2ZS1hJyA6ICcnKTsKICAgIHBpbGwyLnRleHRDb250ZW50ID0gbC5mbGFnICsgJyAnICsgbC5uYW1lOwogICAgcGlsbDIub25jbGljayA9ICgpID0+IHsKICAgICAgZ3Vlc3RIZWFyTGFuZyA9IGwuY29kZTsKICAgICAgZG9jdW1lbnQucXVlcnlTZWxlY3RvckFsbCgnI2d1ZXN0LWhlYXItbGFuZ3MgLmxhbmctcGlsbCcpLmZvckVhY2gocCA9PiBwLmNsYXNzTmFtZSA9ICdsYW5nLXBpbGwnKTsKICAgICAgcGlsbDIuY2xhc3NOYW1lID0gJ2xhbmctcGlsbCBhY3RpdmUtYSc7CiAgICAgIHZhbGlkYXRlTGFuZ3MoKTsKICAgIH07CiAgICBoZWFyQ29udGFpbmVyLmFwcGVuZENoaWxkKHBpbGwyKTsKICB9KTsKfQoKZnVuY3Rpb24gdmFsaWRhdGVMYW5ncygpIHsKICBjb25zdCB3YXJuID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2xhbmctd2FybmluZycpOwogIGNvbnN0IGJ0biA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdidG4tam9pbicpOwogIGlmIChndWVzdFNwZWFrTGFuZyA9PT0gZ3Vlc3RIZWFyTGFuZykgewogICAgd2Fybi5zdHlsZS5kaXNwbGF5ID0gJ2Jsb2NrJzsKICAgIGJ0bi5kaXNhYmxlZCA9IHRydWU7CiAgfSBlbHNlIHsKICAgIHdhcm4uc3R5bGUuZGlzcGxheSA9ICdub25lJzsKICAgIGJ0bi5kaXNhYmxlZCA9IGZhbHNlOwogIH0KfQoKLy8g4pSA4pSAIEpPSU4gU0VTU0lPTiDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2J0bi1qb2luJykuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCBhc3luYyAoKSA9PiB7CiAgLy8gUmVxdWVzdCBtaWMgcGVybWlzc2lvbgogIHRyeSB7CiAgICBhd2FpdCBuYXZpZ2F0b3IubWVkaWFEZXZpY2VzLmdldFVzZXJNZWRpYSh7IGF1ZGlvOiB0cnVlIH0pOwogIH0gY2F0Y2ggKGUpIHsKICAgIHNob3dOb3RpZign4p2MIE1pY3JvcGhvbmUgcGVybWlzc2lvbiBkZW5pZWQuIFBsZWFzZSBhbGxvdyBhY2Nlc3MuJyk7CiAgICByZXR1cm47CiAgfQogIGJ1aWxkU2Vzc2lvblNjcmVlbigpOwogIHNob3dTY3JlZW4oJ3Nlc3Npb24nKTsKICBjb25uZWN0V2ViU29ja2V0KCk7Cn0pOwoKZnVuY3Rpb24gYnVpbGRTZXNzaW9uU2NyZWVuKCkgewogIGNvbnN0IHNwZWFrTCA9IGdldExhbmcoZ3Vlc3RTcGVha0xhbmcpOwogIGNvbnN0IGhlYXJMID0gZ2V0TGFuZyhndWVzdEhlYXJMYW5nKTsKICBjb25zdCByb3cgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnbGFuZy1pbmZvLXJvdycpOwogIHJvdy5pbm5lckhUTUwgPSBgCiAgICA8ZGl2IGNsYXNzPSJsYW5nLWluZm8tY2FyZCIgc3R5bGU9ImJvcmRlci1jb2xvcjpyZ2JhKDE5NiwxODEsMjUzLDAuMykiPgogICAgICA8ZGl2IGNsYXNzPSJsYW5nLWluZm8taWNvbiI+8J+OmTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJsYW5nLWluZm8tbGFiZWwiIHN0eWxlPSJjb2xvcjp2YXIoLS1wdXJwbGUtbGlnaHQpIj5Zb3Ugc3BlYWs8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibGFuZy1pbmZvLW5hbWUiPiR7c3BlYWtMLmZsYWd9ICR7c3BlYWtMLm5hbWV9PC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImxhbmctYXJyb3ciPuKGkjwvZGl2PgogICAgPGRpdiBjbGFzcz0ibGFuZy1pbmZvLWNhcmQiIHN0eWxlPSJib3JkZXItY29sb3I6cmdiYSgxMjksMTQwLDI0OCwwLjMpIj4KICAgICAgPGRpdiBjbGFzcz0ibGFuZy1pbmZvLWljb24iPvCfjqc8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibGFuZy1pbmZvLWxhYmVsIiBzdHlsZT0iY29sb3I6dmFyKC0tcHVycGxlKSI+WW91IGhlYXI8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ibGFuZy1pbmZvLW5hbWUiPiR7aGVhckwuZmxhZ30gJHtoZWFyTC5uYW1lfTwvZGl2PgogICAgPC9kaXY+CiAgYDsKfQoKLy8g4pSA4pSAIFdFQlNPQ0tFVCDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKZnVuY3Rpb24gY29ubmVjdFdlYlNvY2tldCgpIHsKICBzZXRTdGF0dXMoJ0Nvbm5lY3RpbmcuLi4nLCAnY29ubmVjdGluZycpOwogIHdzID0gbmV3IFdlYlNvY2tldChgJHtCQUNLRU5EX1dTfT9yb29tSWQ9JHtyb29tSWR9JnJvbGU9Z3Vlc3RgKTsKCiAgd3Mub25vcGVuID0gKCkgPT4gewogICAgY29uc29sZS5sb2coJ1tXU10gQ29ubmVjdGVkJyk7CiAgICBwaW5nSW50ZXJ2YWwgPSBzZXRJbnRlcnZhbCgoKSA9PiB7IGlmICh3cyAmJiB3cy5yZWFkeVN0YXRlID09PSBXZWJTb2NrZXQuT1BFTikgd3Muc2VuZChKU09OLnN0cmluZ2lmeSh7IHR5cGU6ICdwaW5nJyB9KSk7IH0sIDI1MDAwKTsKICB9OwoKICB3cy5vbm1lc3NhZ2UgPSAoZSkgPT4gewogICAgdHJ5IHsKICAgICAgY29uc3QgbXNnID0gSlNPTi5wYXJzZShlLmRhdGEpOwogICAgICBoYW5kbGVXc01lc3NhZ2UobXNnKTsKICAgIH0gY2F0Y2ggKGVycikgeyBjb25zb2xlLmxvZygnW1dTXSBQYXJzZSBlcnJvcicsIGVycik7IH0KICB9OwoKICB3cy5vbmNsb3NlID0gKGUpID0+IHsKICAgIGNsZWFySW50ZXJ2YWwocGluZ0ludGVydmFsKTsKICAgIGlmIChlLmNvZGUgPT09IDQwMDQpIHsKICAgICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2VuZGVkLXJlYXNvbicpLnRleHRDb250ZW50ID0gJ1Jvb20gbm90IGZvdW5kIG9yIGV4cGlyZWQuJzsKICAgICAgc2hvd1NjcmVlbignZW5kZWQnKTsKICAgIH0KICB9OwoKICB3cy5vbmVycm9yID0gKCkgPT4gc2V0U3RhdHVzKCdDb25uZWN0aW9uIGVycm9yJywgJycpOwp9CgpmdW5jdGlvbiBoYW5kbGVXc01lc3NhZ2UobXNnKSB7CiAgc3dpdGNoIChtc2cudHlwZSkgewogICAgY2FzZSAnY29ubmVjdGVkJzoKICAgICAgc2V0U3RhdHVzKCdXYWl0aW5nIGZvciBob3N0Li4uJywgJ2Nvbm5lY3RpbmcnKTsKICAgICAgYnJlYWs7CiAgICBjYXNlICdwZWVyX2pvaW5lZCc6CiAgICAgIGlmIChtc2cucm9sZSA9PT0gJ2hvc3QnKSB7CiAgICAgICAgc2V0U3RhdHVzKCdIb3N0IGNvbm5lY3RlZCDigJQgcmVhZHknLCAnYWN0aXZlJyk7CiAgICAgICAgc2hvd05vdGlmKCfwn46JIEhvc3QgY29ubmVjdGVkIScpOwogICAgICB9CiAgICAgIGJyZWFrOwogICAgY2FzZSAncGVlcl9sZWZ0JzoKICAgICAgaWYgKG1zZy5yb2xlID09PSAnaG9zdCcpIHsKICAgICAgICBzZXRTdGF0dXMoJ0hvc3QgZGlzY29ubmVjdGVkJywgJycpOwogICAgICAgIHNob3dOb3RpZign4pqg77iPIEhvc3QgbGVmdCB0aGUgc2Vzc2lvbicpOwogICAgICAgIHN0b3BSZWNvcmRpbmcoKTsKICAgICAgICBzZXRUaW1lb3V0KCgpID0+IHsKICAgICAgICAgIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdlbmRlZC1yZWFzb24nKS50ZXh0Q29udGVudCA9ICdUaGUgaG9zdCBoYXMgbGVmdCB0aGUgc2Vzc2lvbi4nOwogICAgICAgICAgc2hvd1NjcmVlbignZW5kZWQnKTsKICAgICAgICB9LCAyMDAwKTsKICAgICAgfQogICAgICBicmVhazsKICAgIGNhc2UgJ3RyYW5zbGF0aW9uJzoKICAgICAgLy8gUmVjaWJpbW9zIHRyYWR1Y2Npw7NuIGRlbCBob3N0IOKGkiByZXByb2R1Y2lyIGNvbiBUVFMKICAgICAgYWRkQnViYmxlKCdob3N0JywgbXNnLm9yaWdpbmFsLCBtc2cudHJhbnNsYXRlZCwgbXNnLnNyY0xhbmcsIG1zZy50Z3RMYW5nKTsKICAgICAgaWYgKGlzQWN0aXZlKSB7CiAgICAgICAgcmVwbGF5VHRzKG1zZy50cmFuc2xhdGVkLCBtc2cudGd0TGFuZyk7CiAgICAgIH0KICAgICAgYnJlYWs7CiAgICBjYXNlICdyb29tX2Nsb3NlZCc6CiAgICAgIHN0b3BSZWNvcmRpbmcoKTsKICAgICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2VuZGVkLXJlYXNvbicpLnRleHRDb250ZW50ID0gbXNnLnJlYXNvbiA9PT0gJ2luYWN0aXZpdHknCiAgICAgICAgPyAnU2Vzc2lvbiBjbG9zZWQgZHVlIHRvIGluYWN0aXZpdHkgKDUgbWluKS4nCiAgICAgICAgOiAnVGhlIHNlc3Npb24gaGFzIGJlZW4gY2xvc2VkIGJ5IHRoZSBob3N0Lic7CiAgICAgIHNob3dTY3JlZW4oJ2VuZGVkJyk7CiAgICAgIGJyZWFrOwogICAgY2FzZSAncG9uZyc6CiAgICAgIGJyZWFrOwogIH0KfQoKLy8g4pSA4pSAIFNUQVJUIC8gU1RPUCDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2J0bi1zdGFydCcpLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgc3RhcnRTZXNzaW9uKTsKZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2J0bi1zdG9wJykuYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCBzdG9wU2Vzc2lvbik7Cgphc3luYyBmdW5jdGlvbiBzdGFydFNlc3Npb24oKSB7CiAgY29uc3Qgb2sgPSBhd2FpdCByZXF1ZXN0TWljKCk7CiAgaWYgKCFvaykgcmV0dXJuOwogIGlzQWN0aXZlID0gdHJ1ZTsKICBjaHVua0NvdW50ID0gMDsKICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYnRuLXN0YXJ0Jykuc3R5bGUuZGlzcGxheSA9ICdub25lJzsKICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYnRuLXN0b3AnKS5zdHlsZS5kaXNwbGF5ID0gJ2Jsb2NrJzsKICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmVjb3JkaW5nLWJhcicpLnN0eWxlLmRpc3BsYXkgPSAnZmxleCc7CiAgc2V0U3RhdHVzKCdBY3RpdmUnLCAnYWN0aXZlJyk7CiAgcmVjb3JkQ2h1bmsoKTsKfQoKZnVuY3Rpb24gc3RvcFNlc3Npb24oKSB7CiAgaXNBY3RpdmUgPSBmYWxzZTsKICBzdG9wUmVjb3JkaW5nKCk7CiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2J0bi1zdGFydCcpLnN0eWxlLmRpc3BsYXkgPSAnYmxvY2snOwogIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdidG4tc3RvcCcpLnN0eWxlLmRpc3BsYXkgPSAnbm9uZSc7CiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlY29yZGluZy1iYXInKS5zdHlsZS5kaXNwbGF5ID0gJ25vbmUnOwogIHNldFN0YXR1cygnUGF1c2VkJywgJycpOwp9Cgphc3luYyBmdW5jdGlvbiByZXF1ZXN0TWljKCkgewogIHRyeSB7CiAgICBjb25zdCBzdHJlYW0gPSBhd2FpdCBuYXZpZ2F0b3IubWVkaWFEZXZpY2VzLmdldFVzZXJNZWRpYSh7IGF1ZGlvOiB0cnVlLCB2aWRlbzogZmFsc2UgfSk7CiAgICBzdHJlYW0uZ2V0VHJhY2tzKCkuZm9yRWFjaCh0ID0+IHQuc3RvcCgpKTsgLy8gc29sbyB2ZXJpZmljYXIgcGVybWlzbwogICAgcmV0dXJuIHRydWU7CiAgfSBjYXRjaCAoZSkgewogICAgc2hvd05vdGlmKCfinYwgTWljcm9waG9uZSBwZXJtaXNzaW9uIGRlbmllZC4nKTsKICAgIHJldHVybiBmYWxzZTsKICB9Cn0KCi8vIOKUgOKUgCBSRUNPUkRJTkcgTE9PUCDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKYXN5bmMgZnVuY3Rpb24gcmVjb3JkQ2h1bmsoKSB7CiAgaWYgKCFpc0FjdGl2ZSkgcmV0dXJuOwoKICB0cnkgewogICAgY29uc3Qgc3RyZWFtID0gYXdhaXQgbmF2aWdhdG9yLm1lZGlhRGV2aWNlcy5nZXRVc2VyTWVkaWEoeyBhdWRpbzogdHJ1ZSwgdmlkZW86IGZhbHNlIH0pOwoKICAgIC8vIE5pdmVsIGRlIGF1ZGlvIHZpc3VhbAogICAgYXVkaW9Db250ZXh0ID0gbmV3IEF1ZGlvQ29udGV4dCgpOwogICAgYW5hbHlzZXIgPSBhdWRpb0NvbnRleHQuY3JlYXRlQW5hbHlzZXIoKTsKICAgIGNvbnN0IHNvdXJjZSA9IGF1ZGlvQ29udGV4dC5jcmVhdGVNZWRpYVN0cmVhbVNvdXJjZShzdHJlYW0pOwogICAgc291cmNlLmNvbm5lY3QoYW5hbHlzZXIpOwogICAgYW5hbHlzZXIuZmZ0U2l6ZSA9IDI1NjsKICAgIGNvbnN0IGRhdGFBcnJheSA9IG5ldyBVaW50OEFycmF5KGFuYWx5c2VyLmZyZXF1ZW5jeUJpbkNvdW50KTsKICAgIGZ1bmN0aW9uIHVwZGF0ZUxldmVsKCkgewogICAgICBpZiAoIWlzQWN0aXZlKSByZXR1cm47CiAgICAgIGFuYWx5c2VyLmdldEJ5dGVGcmVxdWVuY3lEYXRhKGRhdGFBcnJheSk7CiAgICAgIGNvbnN0IGF2ZyA9IGRhdGFBcnJheS5yZWR1Y2UoKGEsIGIpID0+IGEgKyBiLCAwKSAvIGRhdGFBcnJheS5sZW5ndGg7CiAgICAgIGNvbnN0IHBjdCA9IE1hdGgubWluKDEwMCwgKGF2ZyAvIDEyOCkgKiAxMDApOwogICAgICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYXVkaW8tbGV2ZWwtZmlsbCcpLnN0eWxlLndpZHRoID0gcGN0ICsgJyUnOwogICAgICBhbmltRnJhbWVJZCA9IHJlcXVlc3RBbmltYXRpb25GcmFtZSh1cGRhdGVMZXZlbCk7CiAgICB9CiAgICB1cGRhdGVMZXZlbCgpOwoKICAgIC8vIEFjdHVhbGl6YXIgaW5kaWNhZG9yCiAgICBjb25zdCBkb3QgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmVjb3JkaW5nLWRvdCcpOwogICAgY29uc3QgdHh0ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlY29yZGluZy10ZXh0Jyk7CiAgICBkb3Quc3R5bGUuYmFja2dyb3VuZCA9IGNodW5rQ291bnQgPT09IDAgPyAnI0VGNDQ0NCcgOiAnIzM0Qzc1OSc7CiAgICB0eHQudGV4dENvbnRlbnQgPSBjaHVua0NvdW50ID09PSAwCiAgICAgID8gJ/CfjpkgQ2FwdHVyaW5nIGF1ZGlvIOKAlCBmaXJzdCB0cmFuc2xhdGlvbiBpbiB+NHMnCiAgICAgIDogYPCfjpkgUmVjb3JkaW5nIMK3IFRyYW5zbGF0aW9uIGFjdGl2ZSDCtyAke2NodW5rQ291bnR9IGNodW5rc2A7CgogICAgY29uc3QgbWltZVR5cGUgPSBNZWRpYVJlY29yZGVyLmlzVHlwZVN1cHBvcnRlZCgnYXVkaW8vd2VibTtjb2RlY3M9b3B1cycpCiAgICAgID8gJ2F1ZGlvL3dlYm07Y29kZWNzPW9wdXMnIDogJ2F1ZGlvL3dlYm0nOwoKICAgIG1lZGlhUmVjb3JkZXIgPSBuZXcgTWVkaWFSZWNvcmRlcihzdHJlYW0sIHsgbWltZVR5cGUgfSk7CiAgICBjb25zdCBjaHVua3MgPSBbXTsKICAgIG1lZGlhUmVjb3JkZXIub25kYXRhYXZhaWxhYmxlID0gZSA9PiB7IGlmIChlLmRhdGEuc2l6ZSA+IDApIGNodW5rcy5wdXNoKGUuZGF0YSk7IH07CgogICAgbWVkaWFSZWNvcmRlci5vbnN0b3AgPSBhc3luYyAoKSA9PiB7CiAgICAgIC8vIFN0b3AgYXVkaW8gbGV2ZWwgYW5pbWF0aW9uCiAgICAgIGNhbmNlbEFuaW1hdGlvbkZyYW1lKGFuaW1GcmFtZUlkKTsKICAgICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F1ZGlvLWxldmVsLWZpbGwnKS5zdHlsZS53aWR0aCA9ICcwJSc7CiAgICAgIHN0cmVhbS5nZXRUcmFja3MoKS5mb3JFYWNoKHQgPT4gdC5zdG9wKCkpOwogICAgICBpZiAoYXVkaW9Db250ZXh0KSB7IGF1ZGlvQ29udGV4dC5jbG9zZSgpOyBhdWRpb0NvbnRleHQgPSBudWxsOyB9CgogICAgICBpZiAoIWlzQWN0aXZlKSByZXR1cm47CgogICAgICAvLyBTaWd1aWVudGUgY2h1bmsgaW5tZWRpYXRhbWVudGUKICAgICAgc2V0VGltZW91dCgoKSA9PiByZWNvcmRDaHVuaygpLCAwKTsKCiAgICAgIC8vIFByb2Nlc2FyIGVzdGUgY2h1bmsKICAgICAgaWYgKGNodW5rcy5sZW5ndGggPiAwKSB7CiAgICAgICAgY29uc3QgYmxvYiA9IG5ldyBCbG9iKGNodW5rcywgeyB0eXBlOiBtaW1lVHlwZSB9KTsKICAgICAgICBhd2FpdCBwcm9jZXNzQ2h1bmsoYmxvYik7CiAgICAgIH0KICAgIH07CgogICAgbWVkaWFSZWNvcmRlci5zdGFydCgpOwogICAgc2V0VGltZW91dCgoKSA9PiB7IGlmIChtZWRpYVJlY29yZGVyICYmIG1lZGlhUmVjb3JkZXIuc3RhdGUgPT09ICdyZWNvcmRpbmcnKSBtZWRpYVJlY29yZGVyLnN0b3AoKTsgfSwgQ0hVTktfTVMpOwoKICB9IGNhdGNoIChlKSB7CiAgICBjb25zb2xlLmVycm9yKCdSZWNvcmQgZXJyb3I6JywgZSk7CiAgICBpZiAoaXNBY3RpdmUpIHNldFRpbWVvdXQoKCkgPT4gcmVjb3JkQ2h1bmsoKSwgNTAwKTsKICB9Cn0KCmZ1bmN0aW9uIHN0b3BSZWNvcmRpbmcoKSB7CiAgaXNBY3RpdmUgPSBmYWxzZTsKICBjYW5jZWxBbmltYXRpb25GcmFtZShhbmltRnJhbWVJZCk7CiAgaWYgKG1lZGlhUmVjb3JkZXIgJiYgbWVkaWFSZWNvcmRlci5zdGF0ZSA9PT0gJ3JlY29yZGluZycpIHsKICAgIHRyeSB7IG1lZGlhUmVjb3JkZXIuc3RvcCgpOyB9IGNhdGNoIChlKSB7fQogIH0KICBpZiAoYXVkaW9Db250ZXh0KSB7IGF1ZGlvQ29udGV4dC5jbG9zZSgpOyBhdWRpb0NvbnRleHQgPSBudWxsOyB9CiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2F1ZGlvLWxldmVsLWZpbGwnKS5zdHlsZS53aWR0aCA9ICcwJSc7CiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlY29yZGluZy1iYXInKS5zdHlsZS5kaXNwbGF5ID0gJ25vbmUnOwp9Cgphc3luYyBmdW5jdGlvbiBwcm9jZXNzQ2h1bmsoYmxvYikgewogIHRyeSB7CiAgICAvLyAxLiBUcmFuc2NyaWJpcgogICAgY29uc3QgZm9ybURhdGEgPSBuZXcgRm9ybURhdGEoKTsKICAgIGZvcm1EYXRhLmFwcGVuZCgnYXVkaW8nLCBibG9iLCAnY2h1bmsud2VibScpOwogICAgZm9ybURhdGEuYXBwZW5kKCdsYW5ndWFnZScsIGd1ZXN0U3BlYWtMYW5nLnRvTG93ZXJDYXNlKCkuc2xpY2UoMCwyKSk7CgogICAgY29uc3QgdHJhbnNjcmliZVJlcyA9IGF3YWl0IGZldGNoKGAke0JBQ0tFTkRfSFRUUH0vdHJhbnNjcmliZS1yZWFsdGltZWAsIHsKICAgICAgbWV0aG9kOiAnUE9TVCcsIGJvZHk6IGZvcm1EYXRhLAogICAgfSk7CiAgICBjb25zdCB7IHRleHQgPSAnJyB9ID0gYXdhaXQgdHJhbnNjcmliZVJlcy5qc29uKCk7CiAgICBpZiAoIXRleHQudHJpbSgpKSByZXR1cm47CgogICAgLy8gMi4gRW52aWFyIGFsIGJhY2tlbmQgcGFyYSB0cmFkdWNpciB5IHJlbGF5IGFsIGhvc3QKICAgIGNvbnN0IHJlbGF5UmVzID0gYXdhaXQgZmV0Y2goYCR7QkFDS0VORF9IVFRQfS9yb29tL3JlbGF5YCwgewogICAgICBtZXRob2Q6ICdQT1NUJywKICAgICAgaGVhZGVyczogeyAnQ29udGVudC1UeXBlJzogJ2FwcGxpY2F0aW9uL2pzb24nIH0sCiAgICAgIGJvZHk6IEpTT04uc3RyaW5naWZ5KHsgcm9vbUlkLCByb2xlOiAnZ3Vlc3QnLCB0ZXh0OiB0ZXh0LnRyaW0oKSB9KSwKICAgIH0pOwogICAgY29uc3QgeyB0cmFuc2xhdGVkVGV4dCA9ICcnLCByZWxheWVkIH0gPSBhd2FpdCByZWxheVJlcy5qc29uKCk7CiAgICBpZiAoIXRyYW5zbGF0ZWRUZXh0KSByZXR1cm47CgogICAgY2h1bmtDb3VudCsrOwoKICAgIC8vIDMuIE1vc3RyYXIgYnVyYnVqYSBwcm9waWEKICAgIGFkZEJ1YmJsZSgnZ3Vlc3QnLCB0ZXh0LnRyaW0oKSwgdHJhbnNsYXRlZFRleHQsIGd1ZXN0U3BlYWtMYW5nLCBndWVzdEhlYXJMYW5nKTsKCiAgICAvLyBBY3R1YWxpemFyIGluZGljYWRvcgogICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlY29yZGluZy1kb3QnKS5zdHlsZS5iYWNrZ3JvdW5kID0gJyMzNEM3NTknOwogICAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3JlY29yZGluZy10ZXh0JykudGV4dENvbnRlbnQgPQogICAgICBg8J+OmSBSZWNvcmRpbmcgwrcgVHJhbnNsYXRpb24gYWN0aXZlIMK3ICR7Y2h1bmtDb3VudH0gY2h1bmtzYDsKCiAgfSBjYXRjaCAoZSkgewogICAgY29uc29sZS5lcnJvcignUHJvY2VzcyBlcnJvcjonLCBlKTsKICB9Cn0KCi8vIOKUgOKUgCBMRUFWRSAvIFJFUEVBVCDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAKZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2J0bi1sZWF2ZScpLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4gewogIHN0b3BSZWNvcmRpbmcoKTsKICBpZiAod3MpIHdzLmNsb3NlKCk7CiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ2VuZGVkLXJlYXNvbicpLnRleHRDb250ZW50ID0gJ1lvdSBsZWZ0IHRoZSBzZXNzaW9uLic7CiAgc2hvd1NjcmVlbignZW5kZWQnKTsKfSk7Cgpkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnYnRuLXJlcGVhdCcpLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4gewogIGlmIChsYXN0VHJhbnNsYXRpb24pIHJlcGxheVR0cyhsYXN0VHJhbnNsYXRpb24udHJhbnNsYXRlZCwgbGFzdFRyYW5zbGF0aW9uLnR0c0xvY2FsZSk7Cn0pOwoKLy8g4pSA4pSAIFNQRUVEIEJVVFRPTlMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSACmRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5zcGVlZC1idG4nKS5mb3JFYWNoKGJ0biA9PiB7CiAgYnRuLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4gewogICAgdHRzU3BlZWQgPSBwYXJzZUZsb2F0KGJ0bi5kYXRhc2V0LnNwZWVkKTsKICAgIGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJy5zcGVlZC1idG4nKS5mb3JFYWNoKGIgPT4gYi5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKSk7CiAgICBidG4uY2xhc3NMaXN0LmFkZCgnYWN0aXZlJyk7CiAgfSk7Cn0pOwoKLy8g4pSA4pSAIFNUQVJUIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgAppbml0KCk7Cjwvc2NyaXB0Pgo8L2JvZHk+CjwvaHRtbD4K', 'base64').toString('utf-8');

app.get('/join/:roomId', (req, res) => {
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(GUEST_HTML);
});
app.get('/join', (req, res) => {
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(GUEST_HTML);
});

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