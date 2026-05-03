import { GoogleGenAI } from "@google/genai";
import { zodToJsonSchema } from "zod-to-json-schema";
import dotenv from "dotenv";
dotenv.config();

export const ai = new GoogleGenAI({
  vertexai: true,
  project: process.env.GCP_PROJECT_ID,
  location: "global", // GenAI models are available globally, but you can specify a region if needed
});

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

export async function embedText(text) {
  const result = await ai.models.embedContent({
    model: "gemini-embedding-001",
    contents: text,
  });
  return result.embeddings[0].values;
}

export async function embedTexts(texts) {
  const vectors = [];
  for (const text of texts) {
    vectors.push(await embedText(text));
  }
  return vectors;
}

export async function batchEmbedTexts(texts) {
  const result = await ai.models.embedContent({
    model: "gemini-embedding-001",
    contents: texts,
  });
  return result.embeddings.map((e) => e.values);
}
