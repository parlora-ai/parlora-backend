/**
 * backend/index.js
 *
 * Proxy backend para Parlora AI.
 * Despliega en Firebase Cloud Functions o Cloud Run (Node.js 20).
 *
 * Responsabilidades:
 *  - Verificar el JWT de Firebase de cada petición
 *  - Llamar a DeepL API con la clave secreta del servidor
 *  - Llamar a Google Cloud Translation como fallback
 *  - Autenticar al usuario con Google idToken y devolver JWT propio
 *
 * Las claves API NUNCA llegan al cliente.
 * Instalar: npm install express firebase-admin node-fetch dotenv
 */

const express = require('express');
const admin = require('firebase-admin');
const fetch = require('node-fetch');
require('dotenv').config();

const app = express();
app.use(express.json());

// ── Firebase Admin ────────────────────────────────────────────────
admin.initializeApp({
  credential: admin.credential.applicationDefault(),
});

// ── Middleware: verificar JWT de Firebase ─────────────────────────
async function requireAuth(req, res, next) {
  const header = req.headers.authorization ?? '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : null;

  if (!token) return res.status(401).json({ error: 'NO_TOKEN' });

  try {
    req.user = await admin.auth().verifyIdToken(token);
    next();
  } catch {
    return res.status(401).json({ error: 'INVALID_TOKEN' });
  }
}

// ── POST /auth/google ─────────────────────────────────────────────
// Recibe idToken de Google, lo verifica y devuelve un token de Firebase.
app.post('/auth/google', async (req, res) => {
  const { idToken } = req.body;
  if (!idToken) return res.status(400).json({ error: 'NO_ID_TOKEN' });

  try {
    // Verificar el idToken de Google con Firebase Admin
    const decoded = await admin.auth().verifyIdToken(idToken);

    // Crear un custom token de Firebase para el cliente
    const customToken = await admin.auth().createCustomToken(decoded.uid);

    res.json({ token: customToken, uid: decoded.uid });
  } catch (e) {
    console.error('[auth/google]', e);
    res.status(401).json({ error: 'INVALID_GOOGLE_TOKEN' });
  }
});

// ── GET /auth/verify ──────────────────────────────────────────────
// Verifica que el token del cliente sigue siendo válido.
app.get('/auth/verify', requireAuth, (req, res) => {
  res.json({ valid: true, uid: req.user.uid });
});

// ── POST /translate ───────────────────────────────────────────────
// Motor de traducción con DeepL (primario) y Google Cloud (fallback).
app.post('/translate', requireAuth, async (req, res) => {
  const { text, source_lang, target_lang, engine = 'deepl' } = req.body;

  if (!text || !target_lang) {
    return res.status(400).json({ error: 'MISSING_PARAMS' });
  }

  if (engine === 'deepl') {
    // ── DeepL ──
    try {
      const body = new URLSearchParams({
        text,
        target_lang,
        ...(source_lang ? { source_lang } : {}),
      });

      const response = await fetch('https://api-free.deepl.com/v2/translate', {
        method: 'POST',
        headers: {
          Authorization: `DeepL-Auth-Key ${process.env.DEEPL_API_KEY}`,
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: body.toString(),
        // Para el plan Pro: usar api.deepl.com en vez de api-free.deepl.com
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('[deepl] Error:', response.status, errorText);
        return res.status(502).json({ error: 'DEEPL_ERROR', detail: response.status });
      }

      const data = await response.json();
      const translation = data.translations[0];

      return res.json({
        translatedText: translation.text,
        detectedLang: translation.detected_source_language,
        engine: 'deepl',
      });
    } catch (e) {
      console.error('[deepl] Exception:', e);
      return res.status(502).json({ error: 'DEEPL_EXCEPTION' });
    }
  }

  if (engine === 'google') {
    // ── Google Cloud Translation ──
    try {
      const url = `https://translation.googleapis.com/language/translate/v2?key=${process.env.GOOGLE_TRANSLATE_API_KEY}`;

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          q: text,
          target: target_lang.toLowerCase().slice(0, 2), // Google usa 'es', 'en', etc.
          ...(source_lang && source_lang !== 'auto'
            ? { source: source_lang.toLowerCase().slice(0, 2) }
            : {}),
          format: 'text',
        }),
      });

      if (!response.ok) {
        return res.status(502).json({ error: 'GOOGLE_TRANSLATE_ERROR' });
      }

      const data = await response.json();
      const translation = data.data.translations[0];

      return res.json({
        translatedText: translation.translatedText,
        detectedLang: translation.detectedSourceLanguage?.toUpperCase(),
        engine: 'google',
      });
    } catch (e) {
      console.error('[google-translate] Exception:', e);
      return res.status(502).json({ error: 'GOOGLE_TRANSLATE_EXCEPTION' });
    }
  }

  return res.status(400).json({ error: 'UNKNOWN_ENGINE' });
});

// ── Arrancar servidor ─────────────────────────────────────────────
const PORT = process.env.PORT ?? 8080;
app.listen(PORT, () => {
  console.log(`Parlora AI proxy backend escuchando en puerto ${PORT}`);
});

module.exports = app;
