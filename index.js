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
    // Contexto del chunk anterior — DeepL lo usa para coherencia pero NO lo traduce
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

// ── POST /transcribe ──────────────────────────────────────────────
// Recibe un archivo de audio y lo transcribe con Groq Whisper (gratis)
app.post('/transcribe', upload.single('audio'), async (req, res) => {
  const language = req.body.language ?? 'es';
  const filePath = req.file?.path;

  if (!filePath) return res.status(400).json({ error: 'NO_AUDIO', text: '' });

  try {
    // Preparar FormData para Groq API
    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath), {
      filename: 'audio.m4a',
      contentType: 'audio/m4a',
    });
    formData.append('model', 'whisper-large-v3-turbo'); // modelo más rápido de Groq
    formData.append('language', language.slice(0, 2).toLowerCase()); // 'es', 'en', 'fr'...
    formData.append('response_format', 'json');

    // Llamar a Groq API
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

    // Limpiar archivo temporal
    fs.unlink(filePath, () => {});

    if (groqRes.status !== 200) {
      console.error('Groq error:', groqRes.status, groqRes.body);
      return res.status(502).json({ error: 'GROQ_ERROR', text: '' });
    }

    const text = groqRes.body.text ?? '';
    console.log(`Transcribed: "${text.slice(0, 50)}..."`);
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