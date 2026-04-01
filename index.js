const express = require('express');
const https = require('https');
require('dotenv').config();

const app = express();
app.use(express.json());

function httpPost(hostname, path, headers, body) {
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

app.post('/translate', async (req, res) => {
  const { text, target_lang } = req.body;
  if (!text || !target_lang) return res.status(400).json({ error: 'MISSING_PARAMS' });

  try {
    // Nunca forzamos source_lang — dejamos que DeepL detecte automáticamente
    const params = new URLSearchParams({ text, target_lang });

    const result = await httpPost(
      'api-free.deepl.com',
      '/v2/translate',
      {
        'Authorization': `DeepL-Auth-Key ${process.env.DEEPL_API_KEY}`,
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      params.toString()
    );

    if (result.status !== 200) {
      console.error('DeepL error:', result.status, result.body);
      return res.status(502).json({ error: 'DEEPL_ERROR', detail: result.body });
    }

    const translation = result.body.translations[0];
    return res.json({
      translatedText: translation.text,
      detectedLang: translation.detected_source_language,
      engine: 'deepl',
    });
  } catch (e) {
    console.error('DeepL exception:', e.message);
    return res.status(502).json({ error: 'DEEPL_EXCEPTION', message: e.message });
  }
});

app.get('/', (req, res) => res.json({ status: 'Parlora AI backend running ✓' }));

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`Parlora AI backend escuchando en puerto ${PORT}`));
module.exports = app;