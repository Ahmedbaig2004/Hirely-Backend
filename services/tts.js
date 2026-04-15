import { createClient } from "@deepgram/sdk";
import axios from "axios";
import dotenv from "dotenv";
import { normalizeForSpeech } from "./textNormalizer.js";
dotenv.config();

const deepgram = createClient(process.env.DEEPGRAM_API_KEY);
const F5_TTS_URL =
  process.env.F5_TTS_URL ||
  "https://ahmedbaig6512--hirely-voice-engine-f5voiceengine-generate.modal.run";

/**
 * Generate TTS audio for the given text.
 * @param {string} text - The text to speak.
 * @param {"female"|"male"} voice - Voice gender. "female" uses Deepgram, "male" uses F5-TTS.
 * @returns {Promise<{buffer: Buffer, mime: string}|null>}
 */
export async function generateAudio(text, voice = "female") {
  const spokenText = normalizeForSpeech(text);
  try {
    if (voice === "male") {
      // Male voice — F5-TTS Modal endpoint, returns raw audio/wav
      const response = await axios.post(
        F5_TTS_URL,
        { text: spokenText },
        { responseType: "arraybuffer", timeout: 120_000 },
      );
      return { buffer: Buffer.from(response.data), mime: "audio/wav" };
    }

    // Female voice — Deepgram (unchanged)
    const response = await deepgram.speak.request(
      { text: spokenText },
      {
        model: "aura-luna-en",
        encoding: "mp3",
      },
    );

    const stream = await response.getStream();
    const chunks = [];

    if (!stream) throw new Error("No audio stream received");

    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    return { buffer: Buffer.concat(chunks), mime: "audio/mpeg" };
  } catch (error) {
    console.error(`❌ TTS Error (${voice}):`, error.message);
    return null; // Fail gracefully so the app doesn't crash
  }
}
