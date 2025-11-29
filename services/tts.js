import { createClient } from "@deepgram/sdk";
import dotenv from "dotenv";
import { normalizeForSpeech } from "./textNormalizer.js"; // <--- Import
dotenv.config();

const deepgram = createClient(process.env.DEEPGRAM_API_KEY);

export async function generateAudio(text) {
  try {
    // 1. Request Audio from Deepgram
    const spokenText = normalizeForSpeech(text);
    const response = await deepgram.speak.request(
      { text: spokenText },
      {
        model: "aura-luna-en", // "Asteria" (Female) or "aura-orion-en" (Male)
        encoding: "mp3",
      }
    );

    // 2. Convert Stream to Buffer
    const stream = await response.getStream();
    const chunks = [];

    if (!stream) throw new Error("No audio stream received");

    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    return Buffer.concat(chunks);
  } catch (error) {
    console.error("❌ TTS Error:", error.message);
    return null; // Fail gracefully so the app doesn't crash
  }
}
