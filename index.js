const express = require('express');
const https = require('https');
const http = require('http');
const multer = require('multer');
const fs = require('fs');
const FormData = require('form-data');
require('dotenv').config();

const app = express();
app.use(express.json());
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
// Umbrales calibrados para conferencia real
const NO_SPEECH_THRESHOLD = 0.6;   // prob de silencio — por encima descartamos
const AVG_LOGPROB_THRESHOLD = -0.8; // confianza — por debajo descartamos

// Frases de alucinación conocidas de Whisper (case-insensitive)
const HALLUCINATION_PHRASES = [
  'suscríbete', 'subscribe', 'like y suscríbete', 'gracias por ver',
  'thanks for watching', 'thank you for watching', 'don\'t forget to',
  'no olvides', 'síguenos', 'follow us', 'visit our website',
  'www.', '.com', '.es', 'youtube', 'instagram', 'twitter',
];

function isHallucination(text, segments) {
  if (!text || !text.trim()) return true;

  const lower = text.toLowerCase().trim();

  // Comprobar frases conocidas de alucinación
  if (HALLUCINATION_PHRASES.some(phrase => lower.includes(phrase))) {
    console.log(`[FILTER] Hallucination phrase detected: "${text}"`);
    return true;
  }

  // Si hay segmentos con metadatos, comprobar confianza
  if (segments && segments.length > 0) {
    const avgNoSpeech = segments.reduce((sum, s) => sum + (s.no_speech_prob || 0), 0) / segments.length;
    const avgLogprob = segments.reduce((sum, s) => sum + (s.avg_logprob || 0), 0) / segments.length;

    if (avgNoSpeech > NO_SPEECH_THRESHOLD) {
      console.log(`[FILTER] High no_speech_prob: ${avgNoSpeech.toFixed(2)} — discarding: "${text}"`);
      return true;
    }

    if (avgLogprob < AVG_LOGPROB_THRESHOLD) {
      console.log(`[FILTER] Low avg_logprob: ${avgLogprob.toFixed(2)} — discarding: "${text}"`);
      return true;
    }
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
    formData.append('file', fs.createReadStream(filePath), {
      filename: 'audio.m4a',
      contentType: 'audio/m4a',
    });
    formData.append('model', 'whisper-large-v3-turbo');
    formData.append('language', language.slice(0, 2).toLowerCase());
    // verbose_json devuelve segmentos con no_speech_prob y avg_logprob
    formData.append('response_format', 'verbose_json');

    const groqRes = await new Promise((resolve, reject) => {
      const options = {
        hostname: 'api.groq.com',
        path: '/openai/v1/audio/transcriptions',
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
          ...formData.getHeaders(),
        },
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

    if (groqRes.status !== 200) {
      console.error('Groq error:', groqRes.status, groqRes.body);
      return res.status(502).json({ error: 'GROQ_ERROR', text: '' });
    }

    const text = groqRes.body.text ?? '';
    const segments = groqRes.body.segments ?? [];

    // Aplicar filtro de alucinaciones
    if (isHallucination(text, segments)) {
      return res.json({ text: '', engine: 'groq-whisper', filtered: true });
    }

    console.log(`Transcribed: "${text.slice(0, 80)}"`);
    return res.json({ text, engine: 'groq-whisper' });

  } catch (e) {
    console.error('Transcribe exception:', e.message);
    if (filePath && fs.existsSync(filePath)) fs.unlink(filePath, () => {});
    return res.status(500).json({ error: 'TRANSCRIBE_EXCEPTION', text: '' });
  }
});

// ── GET / ─────────────────────────────────────────────────────────
app.get('/', (req, res) => res.json({ status: 'Parlora AI backend running ✓' }));

// ── Arrancar ──────────────────────────────────────────────────────
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`Parlora AI backend escuchando en puerto ${PORT}`));
module.exports = app;