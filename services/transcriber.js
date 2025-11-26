import Groq from "groq-sdk";
import fs from "fs";
import path from "path";
import os from "os";
import dotenv from "dotenv";
dotenv.config();

// Initialize GroqS
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// Dictionary logic (Keep this! It's still useful)
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

export async function transcribeAudio(audioBuffer) {
  console.log("☁️ Sending audio to Groq Cloud...");

  // 1. Save buffer to a temp file (Groq SDK needs a file stream)
  const tempFilePath = path.join(os.tmpdir(), `upload_${Date.now()}.webm`); // Groq accepts webm directly!
  fs.writeFileSync(tempFilePath, audioBuffer);

  try {
    // 2. Call API
    const transcription = await groq.audio.transcriptions.create({
      file: fs.createReadStream(tempFilePath),
      model: "whisper-large-v3", // The Smartest Model (Better than Medium)
      response_format: "json",
      language: "en",
      temperature: 0.0,
    });

    // 3. Clean & Return
    const cleanText = correctTechnicalTerms(transcription.text);
    console.log(`✅ Transcript: "${cleanText}"`);
    return cleanText;
  } catch (error) {
    console.error("❌ Groq Error:", error.message);
    throw new Error("Transcription failed");
  } finally {
    // 4. Cleanup
    if (fs.existsSync(tempFilePath)) fs.unlinkSync(tempFilePath);
  }
}
