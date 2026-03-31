const express = require('express');
const fetch = require('node-fetch');
require('dotenv').config();

const app = express();
app.use(express.json());

// ── POST /translate ───────────────────────────────────────────────
app.post('/translate', async (req, res) => {
  const { text, source_lang, target_lang, engine = 'deepl' } = req.body;
  if (!text || !target_lang) return res.status(400).json({ error: 'MISSING_PARAMS' });

  if (engine === 'deepl') {
    try {
      const body = new URLSearchParams({ text, target_lang });
      if (source_lang && source_lang !== 'auto') body.append('source_lang', source_lang);

      const response = await fetch('https://api-free.deepl.com/v2/translate', {
        method: 'POST',
        headers: {
          Authorization: `DeepL-Auth-Key ${process.env.DEEPL_API_KEY}`,
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: body.toString(),
      });

      if (!response.ok) {
        const err = await response.text();
        console.error('DeepL error:', err);
        return res.status(502).json({ error: 'DEEPL_ERROR' });
      }

      const data = await response.json();
      const translation = data.translations[0];
      return res.json({
        translatedText: translation.text,
        detectedLang: translation.detected_source_language,
        engine: 'deepl',
      });
    } catch (e) {
      console.error('DeepL exception:', e);
      return res.status(502).json({ error: 'DEEPL_EXCEPTION' });
    }
  }

  return res.status(400).json({ error: 'UNKNOWN_ENGINE' });
});

// ── Health check ──────────────────────────────────────────────────
app.get('/', (req, res) => res.json({ status: 'Parlora AI backend running' }));

// ── Arrancar ──────────────────────────────────────────────────────
const PORT = process.env.PORT ?? 8080;
app.listen(PORT, () => console.log(`Parlora AI backend escuchando en puerto ${PORT}`));
module.exports = app;