import Groq from "groq-sdk";
import fs from "fs";
import path from "path";
import os from "os";
import dotenv from "dotenv";
import { isRomanUrdu, toRomanUrdu } from "./translator.js";
dotenv.config();

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// Urdu/Arabic Unicode block
const URDU_SCRIPT = /[\u0600-\u06FF]/;
// Cyrillic, CJK, Japanese, Korean — indicates hallucination
const HALLUCINATION_SCRIPT = /[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]/;

function correctTechnicalTerms(text) {
  let corrected = text;
  const dictionary = {
    "a stand": "Zustand",
    "the stand": "Zustand",
    "read ducks": "Redux",
    sequel: "SQL",
    "no sequel": "NoSQL",
    "note js": "Node.js",
    "type script": "TypeScript",
  };
  for (const [wrong, right] of Object.entries(dictionary)) {
    const regex = new RegExp(`\\b${wrong}\\b`, "gi");
    corrected = corrected.replace(regex, right);
  }
  return corrected;
}

/**
 * Transcribe audio with automatic language detection.
 * Returns { transcript, language } where language is "en" or "ur".
 *
 * Routing logic:
 *   Urdu/Arabic script  → convert to Roman Urdu via Gemini → language: "ur"
 *   Cyrillic/CJK script → hallucination → retry with language:"en" → language: "en"
 *   Latin + Urdu markers → already Roman Urdu → language: "ur"
 *   Latin, no markers   → English → language: "en"
 */
export async function transcribeAudio(audioBuffer) {
  console.log("☁️ Sending audio to Groq Cloud...");

  const tempFilePath = path.join(os.tmpdir(), `upload_${Date.now()}.webm`);
  fs.writeFileSync(tempFilePath, audioBuffer);

  try {
    // Pass 1: auto-detect language (no language lock)
    const transcription = await groq.audio.transcriptions.create({
      file: fs.createReadStream(tempFilePath),
      model: "whisper-large-v3",
      response_format: "json",
      temperature: 0.0,
    });

    const rawText = transcription.text?.trim() || "";

    // Case 1: Urdu/Arabic script → convert to Roman Urdu
    if (URDU_SCRIPT.test(rawText)) {
      console.log(`🔤 Urdu script detected — converting to Roman Urdu...`);
      const romanUrdu = await toRomanUrdu(rawText);
      const transcript = correctTechnicalTerms(romanUrdu);
      console.log(`✅ Transcript (Roman Urdu): "${transcript}"`);
      return { transcript, language: "ur" };
    }

    // Case 2: Hallucination (Cyrillic, CJK, etc.) → retry with English lock
    if (HALLUCINATION_SCRIPT.test(rawText) || rawText.length < 3) {
      console.log(`⚠️ Hallucination detected ("${rawText}") — retrying with language:"en"...`);
      const fallback = await groq.audio.transcriptions.create({
        file: fs.createReadStream(tempFilePath),
        model: "whisper-large-v3",
        response_format: "json",
        language: "en",
        temperature: 0.0,
      });
      const transcript = correctTechnicalTerms(fallback.text?.trim() || "");
      console.log(`✅ Transcript (English fallback): "${transcript}"`);
      return { transcript, language: "en" };
    }

    // Case 3: Latin script — check for Roman Urdu markers
    const transcript = correctTechnicalTerms(rawText);
    if (isRomanUrdu(rawText)) {
      console.log(`✅ Transcript (Roman Urdu): "${transcript}"`);
      return { transcript, language: "ur" };
    }

    // Case 4: English
    console.log(`✅ Transcript (English): "${transcript}"`);
    return { transcript, language: "en" };

  } catch (error) {
    console.error("❌ Groq Error:", error.message);
    throw new Error("Transcription failed");
  } finally {
    if (fs.existsSync(tempFilePath)) fs.unlinkSync(tempFilePath);
  }
}
