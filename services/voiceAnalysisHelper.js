import { prisma } from "../config/db.js";
import { Redis } from "@upstash/redis";
import dotenv from "dotenv";

dotenv.config();

const redisClient = new Redis({
  url: process.env.REDIS_URL,
  token: process.env.REDIS_TOKEN,
});

/**
 * THE INSIGHT ENGINE (v2.0 — SHAP-driven)
 *
 * Primary path: reads shapExplanations written by the Python service.
 * SHAP tells us exactly which acoustic features drove the model's prediction
 * for this specific turn — so insights reflect actual model reasoning,
 * not hardcoded threshold rules.
 *
 * Fallback path: if shapExplanations is absent (old sessions pre-v2.0),
 * falls back to manual threshold rules so old reports still render correctly.
 */
export function generateVoiceInsights(dbRecord) {
  if (!dbRecord) return { label: "N/A", insights: [], metrics: {} };

  const rawFeatures = dbRecord.rawFeatures || {};
  const shapExplanations = rawFeatures.shapExplanations || [];
  const confidenceScore = dbRecord.confidenceLevel || 0;
  const confidencePct = Math.round(confidenceScore * 100);
  const label = dbRecord.confidenceLabelText || "Unknown";

  // ── SHAP path (v2.0+) ─────────────────────────────────────────────────
  if (shapExplanations.length > 0) {
    // Separate into what helped vs. what hurt, take top 2 of each
    const positive = shapExplanations
      .filter((e) => e.direction === "increased")
      .slice(0, 2);
    const negative = shapExplanations
      .filter((e) => e.direction === "decreased")
      .slice(0, 2);

    // Build insights: strengths first, then areas to work on (max 4)
    const insights = [
      ...positive.map((e) => e.explanation).filter(Boolean),
      ...negative.map((e) => e.explanation).filter(Boolean),
    ].slice(0, 4);

    return {
      label,
      confidencePct,
      insights,
      metrics: {
        topStrength: positive[0]?.label || null,
        topWeakness: negative[0]?.label || null,
        confidenceScore,
        modelVersion: dbRecord.modelVersion || "v2.0",
      },
    };
  }

  // ── Fallback: manual rules (pre-v2.0 sessions) ────────────────────────
  const wpm = dbRecord.wordsPerMinute || 0;
  const jitter = dbRecord.jitter || 0;
  const shimmer = dbRecord.shimmer || 0;
  const pauseRatio = dbRecord.pauseRatio || 0;
  const pitchRange = rawFeatures.pitch_range || 0;
  const energy = dbRecord.energyLevel || 0;

  const strengths = [];
  const improvements = [];

  if (wpm > 0) {
    if (wpm > 160)
      improvements.push(
        `Your pace was ${Math.round(wpm)} WPM, faster than ideal (100-160). Try pausing briefly between key points.`,
      );
    else if (wpm < 100)
      improvements.push(
        `Your pace was ${Math.round(wpm)} WPM, a bit slow. Speaking slightly faster sounds more decisive.`,
      );
    else
      strengths.push(
        `Good speaking pace at ${Math.round(wpm)} WPM — natural and easy to follow.`,
      );
  }

  if (jitter >= 0.02 || shimmer >= 0.08)
    improvements.push(
      "Some vocal unsteadiness detected. Taking a deep breath before answering helps steady your voice.",
    );
  else if (jitter > 0 || shimmer > 0)
    strengths.push("Your voice was steady throughout — this projects composure.");

  if (pitchRange > 0) {
    if (pitchRange > 250)
      strengths.push(
        "Good vocal variety, which keeps the interviewer engaged and shows enthusiasm.",
      );
    else
      improvements.push(
        "Your tone was relatively flat. Varying pitch on key points makes you sound more passionate.",
      );
  }

  if (energy > 0) {
    if (energy > 0.05) strengths.push("Good vocal projection — clear and present.");
    else
      improvements.push(
        "Your voice was quiet. Speaking a bit louder sounds more authoritative.",
      );
  }

  if (pauseRatio > 0.3)
    improvements.push(
      `About ${Math.round(pauseRatio * 100)}% of your response was silence. Preparing a mental outline helps bridge gaps.`,
    );
  else if (pauseRatio > 0.2)
    improvements.push(
      "Some noticeable pauses detected. Bridging phrases like \"that's why...\" can help flow.",
    );
  else if (pauseRatio > 0)
    strengths.push("You spoke fluently with natural pauses — no awkward silences.");

  const insights = [
    ...strengths.slice(0, 2),
    ...improvements.slice(0, 2),
  ];

  return {
    label,
    confidencePct: Math.round((dbRecord.confidenceLevel || 0) * 100),
    insights,
    metrics: {
      pacing: wpm > 160 ? "Fast" : wpm < 100 ? "Slow" : "Ideal",
      stability: jitter >= 0.02 || shimmer >= 0.08 ? "Low" : "High",
      engagement: pitchRange > 250 ? "Dynamic" : "Limited",
      fluency: pauseRatio > 0.2 ? "Hesitant" : "Fluent",
    },
  };
}

/**
 * Save voice analysis results to PostgreSQL via Prisma
 */
export async function saveVoiceAnalysis(result) {
  try {
    return await prisma.voiceAnalysis.create({
      data: {
        interviewTurnId: result.interviewTurnId,
        confidenceLevel: result.confidenceLevel,
        confidenceLabelText: result.confidenceLabelText,
        speakingQuality: result.speakingQuality,
        vocalStability: result.vocalStability,
        speakingFluency: result.speakingFluency,
        pitchMean: result.pitchMean,
        pitchStd: result.pitchStd,
        energyLevel: result.energyLevel,
        wordsPerMinute: result.wordsPerMinute,
        pauseRatio: result.pauseRatio,
        jitter: result.jitter,
        shimmer: result.shimmer,
        modelVersion: result.modelVersion,
        allProbabilities: result.allProbabilities,
        rawFeatures: result.rawFeatures,
        status: result.status,
        errorMessage: result.errorMessage || null,
        processingTimeMs: result.processingTimeMs,
        processedAt: new Date(result.processedAt),
      },
    });
  } catch (e) {
    console.error("Error saving voice analysis:", e);
    throw e;
  }
}

/**
 * Get all voice analyses for an interview and enrich with insights
 */
export async function getVoiceAnalysesForInterview(sessionId, historyCount) {
  try {
    const results = [];

    for (let i = 1; i <= historyCount; i++) {
      const redisKey = `voice_analysis:${sessionId}:${i}`;
      const data = await redisClient.get(redisKey);

      if (data) {
        const parsed = typeof data === "string" ? JSON.parse(data) : data;
        const insights = generateVoiceInsights(parsed);
        results.push({ ...parsed, ...insights });
      }
    }

    return results;
  } catch (e) {
    console.error("Error fetching voice analyses from Redis:", e);
    return [];
  }
}
