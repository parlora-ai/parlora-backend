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
  return crypto.randomBytes(4).toString('hex').toUpperCase(); // ej: A3F7B2C1
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