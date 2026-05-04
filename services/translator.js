import { generate } from "../config/gemini.js";
import dotenv from "dotenv";
dotenv.config();

// Unambiguous Roman Urdu marker words — common function words that won't appear in English
const URDU_MARKERS =
  /\b(hai|hain|tha|thi|kiya|kar|raha|rahi|rahe|nahi|nhi|yeh|woh|kyun|pata|gaya|gayi|mein|mujhe|hum|aap|apko|ka|ki|ke)\b/gi;
const MIN_ROMAN_URDU_MARKERS = 4;

export function isRomanUrdu(text) {
  if (!text) return false;
  const matches = text.match(URDU_MARKERS) || [];
  return matches.length >= MIN_ROMAN_URDU_MARKERS;
}

/**
 * Convert Urdu script (Arabic letters) to Roman Urdu (Latin script).
 * Used when Whisper auto-detects Urdu and outputs in Urdu script.
 *
 * @param {string} urduScriptText - Text in Urdu/Arabic script
 * @returns {Promise<string>} Roman Urdu transliteration
 */
export async function toRomanUrdu(urduScriptText) {
  const prompt = `Convert the following Urdu script text to Roman Urdu (Urdu written in Latin/English letters).
Rules:
- Write each Urdu word phonetically in English letters (e.g. "ہے" → "hai", "میں" → "mein", "نے" → "ne")
- Preserve ALL English technical terms exactly as-is (React, Redux, SQL, API, Node.js, TypeScript, etc.)
- Keep the word order and meaning identical
- Output ONLY the Roman Urdu text, nothing else

Text: "${urduScriptText}"`;

  try {
    const romanUrdu = (
      await generate(prompt, { model: "gemini-3.1-flash-lite-preview" })
    ).trim();
    console.log(`🔤 Urdu script → Roman Urdu`);
    console.log(`   Script:  "${urduScriptText}"`);
    console.log(`   Roman:   "${romanUrdu}"`);
    return romanUrdu;
  } catch (e) {
    console.error("Urdu script → Roman Urdu conversion failed:", e.message);
    return urduScriptText; // fallback: return original script
  }
}

/**
 * Translate Roman Urdu/English mixed text to clean English.
 * Returns the original text unchanged if no Urdu markers are detected.
 *
 * @param {string} text - Transcribed or typed answer
 * @returns {Promise<{ translatedText: string, isTranslated: boolean }>}
 */
export async function translateIfNeeded(text) {
  if (!text || !isRomanUrdu(text)) {
    return { translatedText: text, isTranslated: false };
  }

  const prompt = `Translate the following Roman Urdu/English mixed text to clean English.
Preserve ALL technical terms exactly as-is (React, Redux, SQL, API, Node.js, TypeScript, etc.).
Preserve the meaning and intent precisely. Do not add or remove information.
Output ONLY the English translation, nothing else.

Text: "${text}"`;

  try {
    const translatedText = (
      await generate(prompt, { model: "gemini-2.5-flash-lite" })
    ).trim();
    console.log(`🌐 Urdu detected — translated for evaluation.`);
    console.log(`   Original:    "${text}"`);
    console.log(`   Translated:  "${translatedText}"`);
    return { translatedText, isTranslated: true };
  } catch (e) {
    console.error("Translation failed, using original text:", e.message);
    return { translatedText: text, isTranslated: false };
  }
}
