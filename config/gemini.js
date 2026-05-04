import { GoogleGenAI } from "@google/genai";
import { zodToJsonSchema } from "zod-to-json-schema";
import dotenv from "dotenv";
dotenv.config();

// ─────────────────────────────────────────
// PRIMARY CLIENT  (for generate / chat)
// 2.5-flash & 2.5-flash-lite work best on
// us-central1 or the global endpoint
// ─────────────────────────────────────────
export const ai = new GoogleGenAI({
  vertexai: true,
  project: process.env.GCP_PROJECT_ID,
  location: process.env.GCP_LOCATION || "us-central1",
});

// ─────────────────────────────────────────
// MULTI-REGION EMBEDDING CLIENTS
// Each region has its own independent 5 RPS
// quota → 5 regions = ~25 RPS / ~1,500 RPM
// No daily cap (unlike AI Studio free tier)
// ─────────────────────────────────────────
const EMBED_REGIONS = [
  "us-central1",
  "us-east4",
  "us-east5",
  "us-west1",
  "europe-west4",
];

const embedClients = EMBED_REGIONS.map(
  (location) =>
    new GoogleGenAI({
      vertexai: true,
      project: process.env.GCP_PROJECT_ID,
      location,
    }),
);

let _regionIdx = 0;
function nextEmbedClient() {
  const client = embedClients[_regionIdx % embedClients.length];
  const region = EMBED_REGIONS[_regionIdx % EMBED_REGIONS.length];
  _regionIdx++;
  return { client, region };
}

// ─────────────────────────────────────────
// SCHEMA HELPERS
// ─────────────────────────────────────────
function cleanSchema(schema) {
  if (typeof schema !== "object" || schema === null) return schema;
  const out = {};
  for (const [k, v] of Object.entries(schema)) {
    if (
      [
        "$schema",
        "additionalProperties",
        "$ref",
        "definitions",
        "$defs",
      ].includes(k)
    )
      continue;
    if (k === "properties" && typeof v === "object") {
      out[k] = Object.fromEntries(
        Object.entries(v).map(([pk, pv]) => [pk, cleanSchema(pv)]),
      );
    } else if (k === "items") {
      out[k] = cleanSchema(v);
    } else {
      out[k] = v;
    }
  }
  return out;
}

export function zodToGeminiSchema(zodSchema) {
  return cleanSchema(zodToJsonSchema(zodSchema));
}

// ─────────────────────────────────────────
// GENERATE  (uses primary client, unchanged)
// ─────────────────────────────────────────
export async function generateStructured(prompt, zodSchema, options = {}) {
  const { model = "gemini-2.5-flash", ...rest } = options;
  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config: {
      ...rest,
      responseMimeType: "application/json",
      responseSchema: zodToGeminiSchema(zodSchema),
    },
  });
  return JSON.parse(response.text);
}

export async function generate(prompt, options = {}) {
  const { model = "gemini-2.5-flash", ...rest } = options;
  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config: rest,
  });
  return response.text;
}

// ─────────────────────────────────────────
// EMBED  (rotates across regions)
// ─────────────────────────────────────────

/** Single text — picks next region in rotation */
export async function embedText(text) {
  const { client, region } = nextEmbedClient();
  try {
    const result = await client.models.embedContent({
      model: "gemini-embedding-001",
      contents: text,
      config: { taskType: "RETRIEVAL_DOCUMENT" },
    });
    return result.embeddings[0].values;
  } catch (err) {
    // If this region is throttled, try the next one immediately
    if (
      err?.message?.includes("429") ||
      err?.message?.includes("RESOURCE_EXHAUSTED")
    ) {
      console.warn(`  ⚠️  Region ${region} throttled, falling back to next...`);
      const { client: fallback, region: fbRegion } = nextEmbedClient();
      console.log(`  🔄 Retrying on ${fbRegion}`);
      const result = await fallback.models.embedContent({
        model: "gemini-embedding-001",
        contents: text,
        config: { taskType: "RETRIEVAL_DOCUMENT" },
      });
      return result.embeddings[0].values;
    }
    throw err;
  }
}

/** Multiple texts — each call rotates to the next region */
export async function embedTexts(texts) {
  const vectors = [];
  for (const text of texts) {
    vectors.push(await embedText(text));
  }
  return vectors;
}

/**
 * True batch — sends all texts to ONE region in a single request.
 * Use this when you want throughput over distribution.
 * Still rotates which region handles each batch call.
 */
export async function batchEmbedTexts(texts) {
  const { client, region } = nextEmbedClient();
  console.log(`  🌍 Batch embed → ${region} (${texts.length} texts)`);
  const result = await client.models.embedContent({
    model: "gemini-embedding-001",
    contents: texts,
    config: { taskType: "RETRIEVAL_DOCUMENT" },
  });
  return result.embeddings.map((e) => e.values);
}
