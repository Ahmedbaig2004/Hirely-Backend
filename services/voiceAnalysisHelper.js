import { prisma } from "../config/db.js"; // 👈 CHANGE 1: Use your central adapterimport redis from "redis";

const redisClient = redis.createClient({
  url: process.env.REDIS_URL || "redis://localhost:6379",
});

/**
 * 🆕 THE INSIGHT ENGINE
 * Converts raw acoustic features (from Python/DB) into human-readable feedback.
 */
export function generateVoiceInsights(dbRecord) {
  if (!dbRecord) return { label: "N/A", insights: [], metrics: {} };

  const wpm = dbRecord.wordsPerMinute || 0;
  const jitter = dbRecord.jitter || 0;
  const shimmer = dbRecord.shimmer || 0;
  const pauseRatio = dbRecord.pauseRatio || 0;
  const pitchStd = dbRecord.pitchStd || 0;
  const pitchRange = dbRecord.pitchRange || 0;
  const label = dbRecord.confidenceLabelText || "Unknown";

  const insights = [];

  // 1. Pacing Insight (Based on 130-170 WPM ideal range)
  if (wpm > 190)
    insights.push(
      "Speaking rate is very fast; try to breathe more between sentences.",
    );
  else if (wpm < 110 && wpm > 0)
    insights.push(
      "Speaking rate is slow; may indicate hesitation or deep thinking.",
    );

  // 2. Stability Insight (Scientific Benchmarks: Jitter > 1.5% or Shimmer > 5%)
  if (jitter > 0.015 || shimmer > 0.05) {
    insights.push(
      "Vocal fluctuations detected; this often suggests high-pressure anxiety or nervousness.",
    );
  } else {
    insights.push(
      "Excellent vocal stability; you maintained a very steady and controlled tone.",
    );
  }

  // 3. Engagement & Prosody (Pitch Variance)
  if (pitchRange < 20 && pitchRange > 0) {
    insights.push(
      "Tone is somewhat monotone; try to vary your pitch to sound more engaging.",
    );
  } else if (pitchRange > 50) {
    insights.push("Great vocal range; you sound expressive and natural.");
  }

  // 4. Fluency (Pause Analysis)
  if (pauseRatio > 0.2) {
    insights.push(
      "Higher than average silence detected; work on reducing long hesitations.",
    );
  }

  return {
    label,
    insights,
    metrics: {
      pacing: wpm > 190 ? "Fast" : wpm < 110 ? "Slow" : "Ideal",
      stability: jitter > 0.015 || shimmer > 0.05 ? "Low" : "High",
      engagement: pitchRange < 20 ? "Monotone" : "Dynamic",
      fluency: pauseRatio > 0.2 ? "Hesitant" : "Fluent",
    },
  };
}

/**
 * Save voice analysis results to database
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
 * Get all voice analyses for an interview and include insights
 */
export async function getVoiceAnalysesForInterview(sessionId, historyCount) {
  try {
    const results = [];

    // We loop through the number of turns (e.g., 1 to 9)
    for (let i = 1; i <= historyCount; i++) {
      const redisKey = `voice_analysis:${sessionId}:${i}`;
      const data = await redisClient.get(redisKey); // Use your redis client here

      if (data) {
        const parsedData = JSON.parse(data);
        const insights = generateVoiceInsights(parsedData);
        results.push({ ...parsedData, ...insights });
      }
    }

    return results;
  } catch (e) {
    console.error("Error fetching voice analyses from Redis:", e);
    return [];
  }
}
