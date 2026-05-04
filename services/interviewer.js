import { generateStructured } from "../config/gemini.js";
import { z } from "zod";
import dotenv from "dotenv";
import { getVoiceAnalysesForInterview } from "./voiceAnalysisHelper.js";

dotenv.config();

const W_AUDIO = 0.5;
const W_VIDEO = 0.5;

const NextQuestionSchema = z.object({
  question: z.string(),
  topic: z.string(),
  difficulty: z.string(),
  reason: z.string(),
});

const FinalReportSchema = z.object({
  decision: z.enum(["Strong Hire", "Hire", "Weak Hire", "No Hire"]),
  technicalLevel: z
    .string()
    .describe("Estimated seniority (e.g. Junior, Mid, Senior)"),
  summary: z
    .string()
    .describe("Professional analysis of the candidate's performance"),
  strengths: z.array(z.string()),
  weaknesses: z.array(z.string()),
  recommendations: z
    .string()
    .describe("Advice for both technical and communication growth"),
});

const VoiceInsightsSchema = z.object({
  insights: z
    .array(z.string())
    .describe(
      "4-6 specific, actionable, non-repetitive observations about the candidate's communication style across the full interview",
    ),
});

/**
 * Use AI to synthesize raw voice metrics from all turns into coherent,
 * non-repetitive feedback. One LLM call covers the whole interview.
 */
async function generateAIVoiceInsights(voiceAnalyses) {
  if (!voiceAnalyses || voiceAnalyses.length === 0) return [];

  const turnSummaries = voiceAnalyses.map((v, i) => {
    const shapExplanations = v.rawFeatures?.shapExplanations || [];
    const hasShap = shapExplanations.length > 0;

    const base = {
      turn: i + 1,
      confidenceScore: Math.round((v.confidenceLevel || 0) * 100),
      confidenceLabel: v.confidenceLabelText || "Unknown",
      wpm: Math.round(v.wordsPerMinute || 0),
    };

    if (hasShap) {
      return {
        ...base,
        topFactors: shapExplanations.slice(0, 3).map((e) => ({
          feature: e.label,
          impact: e.direction,
          explanation: e.explanation,
        })),
      };
    }

    return {
      ...base,
      pauseRatioPct: Math.round((v.pauseRatio || 0) * 100),
      jitter: parseFloat((v.jitter || 0).toFixed(3)),
      shimmer: parseFloat((v.shimmer || 0).toFixed(3)),
      energy: parseFloat((v.energyLevel || 0).toFixed(3)),
    };
  });

  const hasShapData = turnSummaries.some((t) => t.topFactors);

  const prompt = `You are an expert communication coach reviewing ML-powered voice analysis from a job interview.

ANALYSIS PER QUESTION:
${JSON.stringify(turnSummaries, null, 2)}

${
  hasShapData
    ? `Each turn includes topFactors — the acoustic features the model identified as most impactful on the confidence score, whether each factor increased or decreased the score, and a coaching tip.`
    : `REFERENCE RANGES: WPM 100-160 ideal. Jitter <0.02 stable. Shimmer <0.08 stable. Pause ratio <20% fluent. Energy >0.05 good projection.`
}

TASK: Write 4-6 concise, actionable observations synthesizing the candidate's communication patterns across the WHOLE interview.

Rules:
- ${hasShapData ? "Use the SHAP-identified factors as your primary evidence — these are what the model actually measured" : "Use the metrics and reference ranges as your evidence"}
- If the same factor appears across multiple turns, treat it as a consistent pattern (stronger evidence)
- If a factor appears positive in some turns and negative in others, note the inconsistency
- Lead with strengths, then areas to improve
- Be specific and actionable (e.g. "try pausing between key points" not "work on pacing")
- Write in second person ("e.g .You maintained steady projection throughout")
- Keep each observation to one sentence
- Do not refer to specific turn numbers`;

  try {
    const result = await generateStructured(prompt, VoiceInsightsSchema);
    return result.insights;
  } catch (e) {
    console.warn("⚠️ AI voice insights generation failed:", e.message);
    return [];
  }
}

/**
 * Adaptive Questioning Logic
 *
 * @param {object} session - Current Redis session
 * @param {object} [currentTurn] - The turn being answered right now { question, answer }.
 *   When provided, question generation runs in parallel with evaluation — Gemini reads
 *   the raw transcript directly instead of waiting for a numeric score.
 */
export async function getNextQuestion(session, currentTurn = null) {
  const historyText = session.history
    .map(
      (h) =>
        `Q: ${h.question}\nA: ${h.answer}\nScore: ${
          h.score == null ? "Not scored yet" : `${h.score}/100`
        }\nDifficulty: ${h.difficulty || "Medium"}\nTopic: ${h.topic || "General"}`,
    )
    .join("\n---\n");

  const lastTurn = session.history[session.history.length - 1];
  const lastDifficulty = lastTurn?.difficulty || "Medium";
  const interviewType = session.interviewType || "JOB_SPECIFIC";
  const config = session.config || {};
  const selectedDifficulty = config.difficulty || "Medium";

  const currentTurnSection = currentTurn
    ? `CURRENT QUESTION (just answered — transcript below, not yet scored):
Q: ${currentTurn.question}
A: ${currentTurn.answer}
Difficulty so far: ${lastDifficulty}

Use this transcript to choose the next question. For fixed-difficulty modes, adapt only the content, not the difficulty.`
    : `LAST QUESTION was difficulty: ${lastDifficulty}. Score is ${
        lastTurn?.score == null ? "not available yet" : `${lastTurn.score}/100`
      }.`;

  let contextSection;
  let taskDescription;

  if (interviewType === "TECHNICAL") {
    const topicsCovered = session.history
      .map((h) => h.topic)
      .filter(Boolean)
      .join(", ");
    contextSection = `INTERVIEW TYPE: Technical — ${config.stack || "General"} (${config.difficulty || "Medium"} level)
Topics already covered: ${topicsCovered || "(none yet)"}`;
    taskDescription = `Generate the NEXT technical question about ${config.stack || "the stack"}.
- Stay at or near "${config.difficulty || "Medium"}" difficulty, adjusting by ONE level based on answer quality.
- Cover a topic NOT already covered above.
- Do NOT ask about the same concept twice.`;
  } else if (interviewType === "BEHAVIORAL") {
    const competenciesCovered = session.history
      .map((h) => h.topic)
      .filter(Boolean)
      .join(", ");
    contextSection = `INTERVIEW TYPE: Behavioral — STAR method (${config.difficulty || "Medium"} level)
Competencies already covered: ${competenciesCovered || "(none yet)"}`;
    taskDescription = `Generate the NEXT behavioral question expecting a STAR-method answer.
- Stay at or near "${config.difficulty || "Medium"}" difficulty, adjusting by ONE level based on answer quality.
- Cover a competency NOT already covered above (e.g. leadership, conflict, failure, prioritization, teamwork).
- Do NOT repeat competencies.`;
  } else {
    contextSection = `JOB CONTEXT: ${(session.jobDescription || "").substring(0, 500)}...`;
    taskDescription = `Generate the NEXT follow-up question. If the candidate showed a gap or said they don't know something, probe that area at an appropriate difficulty. Follow the difficulty rules above strictly.`;
  }

  if (interviewType === "TECHNICAL") {
    const topicsCovered = session.history
      .map((h) => h.topic)
      .filter(Boolean)
      .join(", ");
    contextSection = `INTERVIEW TYPE: Technical - ${config.stack || "General"} (${selectedDifficulty} level)
Topics already covered: ${topicsCovered || "(none yet)"}`;
    taskDescription = `Generate the NEXT technical question about ${config.stack || "the stack"}.
- Keep the difficulty exactly "${selectedDifficulty}". Do not increase or decrease it.
- Adapt by choosing the most useful next topic or follow-up based on the candidate's transcript.
- Cover a topic NOT already covered above.
- Do NOT ask about the same concept twice.
- Set the "difficulty" field to exactly "${selectedDifficulty}".`;
  } else if (interviewType === "BEHAVIORAL") {
    const competenciesCovered = session.history
      .map((h) => h.topic)
      .filter(Boolean)
      .join(", ");
    contextSection = `INTERVIEW TYPE: Behavioral - STAR method (${selectedDifficulty} level)
Competencies already covered: ${competenciesCovered || "(none yet)"}`;
    taskDescription = `Generate the NEXT behavioral question expecting a STAR-method answer.
- Keep the difficulty exactly "${selectedDifficulty}". Do not increase or decrease it.
- Adapt by probing the most useful competency, weakness, missing STAR element, or follow-up area from the candidate's transcript.
- Cover a competency NOT already covered above (e.g. leadership, conflict, failure, prioritization, teamwork).
- Do NOT repeat competencies.
- Set the "difficulty" field to exactly "${selectedDifficulty}".`;
  }

  const modeDifficultyRules =
    interviewType === "TECHNICAL" || interviewType === "BEHAVIORAL"
      ? `FIXED DIFFICULTY OVERRIDE:
    - Ignore any instruction above that says to increase or decrease difficulty.
    - This mode uses the user's selected difficulty, so the next question difficulty must be exactly "${selectedDifficulty}".
    - Adapt only the content: topic, competency, follow-up angle, specificity, or depth within that same difficulty.`
      : `JOB-SPECIFIC DIFFICULTY:
    - Difficulty may increase, decrease, or stay the same based on the candidate's answer quality and the JD requirements.`;

  const prompt = `
    You are a Dynamic Interviewer.
    ${contextSection}

    INTERVIEW HISTORY (previous turns, scored):
    ${historyText || "(No previous turns yet)"}

    ${currentTurnSection}

    ADAPTIVE DIFFICULTY RULES:
    - Strong answer (clear, correct, detailed): you MAY increase difficulty by ONE level (Easy→Medium, Medium→Hard). Never jump two levels.
    - Weak answer (vague, incorrect, "I don't know"): DECREASE difficulty or stay the same. Do NOT keep asking Hard questions after a poor answer.
    - Always set the "difficulty" field to exactly one of: "Easy", "Medium", or "Hard".

    ${modeDifficultyRules}

    TASK: ${taskDescription}
  `;

  return await generateStructured(prompt, NextQuestionSchema);
}

/**
 * Final Report Generation with Integrated Voice Insights
 * @param {string[]} prefetchedVoiceData - Optional raw voice data already fetched from Redis.
 *   When provided, skips a redundant Redis read inside this function.
 */
export async function generateFinalReport(
  sessionId,
  history,
  jobDescription,
  gapAnalysis,
  prefetchedVoiceData = null,
  interviewType = "JOB_SPECIFIC",
  config = null,
  prefetchedVideoData = null,
) {
  // 1. Calculate Technical Score (difficulty-adjusted)
  let roleSeniority = "mid";
  if (interviewType === "JOB_SPECIFIC" && jobDescription) {
    const jdLower = jobDescription.toLowerCase();
    roleSeniority =
      jdLower.includes("senior") ||
      jdLower.includes("lead") ||
      jdLower.includes("principal")
        ? "senior"
        : jdLower.includes("junior") ||
            jdLower.includes("entry") ||
            jdLower.includes("graduate") ||
            jdLower.includes("intern")
          ? "junior"
          : "mid";
  } else if (config?.difficulty) {
    roleSeniority =
      config.difficulty === "Hard"
        ? "senior"
        : config.difficulty === "Easy"
          ? "junior"
          : "mid";
  }

  const difficultyRank = { Easy: 0, Medium: 1, Hard: 2 };
  const roleBaseRank = { junior: 0, mid: 1, senior: 2 };

  function adjustScoreForDifficulty(rawScore, difficulty) {
    const gap = (difficultyRank[difficulty] ?? 1) - roleBaseRank[roleSeniority];
    if (gap > 0) {
      const floored = Math.max(rawScore, 50);
      return rawScore >= 70 ? Math.min(rawScore + 10, 100) : floored;
    }
    return rawScore;
  }

  const technicalScore =
    history.length > 0
      ? history.reduce(
          (sum, turn) =>
            sum +
            adjustScoreForDifficulty(turn.score, turn.difficulty || "Medium"),
          0,
        ) / history.length
      : 0;

  // 1b. Calculate Delivery Score (text modality)
  const turnsWithDelivery = history.filter((t) => t.deliveryAnalysis);
  const deliveryScore =
    turnsWithDelivery.length > 0
      ? turnsWithDelivery.reduce(
          (sum, t) => sum + t.deliveryAnalysis.deliveryScore,
          0,
        ) / turnsWithDelivery.length
      : null;

  // 2. Fetch interpreted Voice Data
  let voiceAnalyses = [];
  try {
    if (prefetchedVoiceData !== null) {
      console.log(`⏳ Using pre-fetched voice data for: ${sessionId}`);
      voiceAnalyses = prefetchedVoiceData;
    } else {
      console.log(`⏳ Extracting vocal insights for: ${sessionId}`);
      voiceAnalyses = await getVoiceAnalysesForInterview(
        sessionId,
        history.length,
      );
    }
  } catch (e) {
    console.warn("⚠️ Voice analysis retrieval failed:", e.message);
  }

  // 3. Aggregate Communication Patterns
  const hasVoiceData = voiceAnalyses.length > 0;
  const voiceScore = hasVoiceData
    ? voiceAnalyses.reduce((sum, v) => sum + v.confidenceLevel * 100, 0) /
      voiceAnalyses.length
    : null;

  console.log(
    `🤖 Generating AI voice insights for ${voiceAnalyses.length} turns...`,
  );
  const allVocalInsights = await generateAIVoiceInsights(voiceAnalyses);

  const deliveryInsights =
    turnsWithDelivery.length > 0
      ? turnsWithDelivery
          .map((t, i) => {
            const d = t.deliveryAnalysis;
            return `Q${i + 1}: Delivery ${d.deliveryScore}/100, Fillers: ${d.fillerCount}, Hedging: ${d.hedgingCount}, Relevance: ${d.relevanceScore}/100, Specificity: ${d.specificityScore}/100. Top improvement: ${d.topImprovement}`;
          })
          .join("\n")
      : "No delivery analysis available.";

  const historyText = history
    .map(
      (h, i) =>
        `Q${i + 1} [${h.difficulty || "Medium"}]: ${h.question}\nTechnical Score: ${h.score}/100`,
    )
    .join("\n\n");

  let interviewContextLine;
  if (interviewType === "TECHNICAL" && config) {
    interviewContextLine = `INTERVIEW TYPE: Technical — ${config.stack || "General"} (${config.difficulty || "Medium"} level, ${config.questionCount || history.length} questions)`;
  } else if (interviewType === "BEHAVIORAL" && config) {
    interviewContextLine = `INTERVIEW TYPE: Behavioral (STAR method, ${config.difficulty || "Medium"} level, ${config.questionCount || history.length} questions)`;
  } else {
    interviewContextLine = `INTERVIEW TYPE: Job-Specific\n    - Resume Gaps: ${JSON.stringify(gapAnalysis)}`;
  }
  const videoAnalyses = prefetchedVideoData || [];
  const hasVideoData = videoAnalyses.length > 0;
  const videoScore = hasVideoData
    ? videoAnalyses.reduce((sum, v) => sum + v.confidenceLevel * 100, 0) /
      videoAnalyses.length
    : null;
  const fusedConfidenceValues = history
    .map((turn) => {
      const S_audio = turn.voiceAnalysis?.confidenceLevel ?? null;
      const S_video = turn.videoAnalysis?.confidenceLevel ?? null;

      if (S_audio !== null && S_video !== null) {
        return W_AUDIO * S_audio + W_VIDEO * S_video;
      }
      if (S_audio !== null) return S_audio;
      if (S_video !== null) return S_video;
      return null;
    })
    .filter((score) => score !== null);
  const fusedScore =
    fusedConfidenceValues.length > 0
      ? parseFloat(
          (
            (fusedConfidenceValues.reduce((sum, score) => sum + score, 0) /
              fusedConfidenceValues.length) *
            100
          ).toFixed(2),
        )
      : null;

  const prompt = `
    You are an expert Technical Hiring Manager.

    ${interviewContextLine}

    TECHNICAL DATA:
    - Score: ${technicalScore.toFixed(1)}/100 (difficulty-adjusted)
    - History: ${historyText}
    - Note: The interview used adaptive difficulty — harder questions were asked when the candidate performed well. A lower score on a Hard question is more acceptable than the same score on an Easy question, especially for junior/entry-level roles.

    COMMUNICATION DATA:
    - Vocal Confidence Score: ${voiceScore !== null ? voiceScore.toFixed(1) + "/100" : "N/A (chat-only interview)"}
    - Vocal Observations: ${allVocalInsights.length > 0 ? allVocalInsights.join("; ") : hasVoiceData ? "Stable and confident delivery." : "No audio data — chat-only interview."}

    DELIVERY ANALYSIS (transcript quality):
    - Overall Delivery Score: ${deliveryScore !== null ? deliveryScore.toFixed(1) + "/100" : "N/A"}
    - Per-question breakdown:
    ${deliveryInsights}

    BODY LANGUAGE DATA:
    - Video Confidence Score: ${videoScore !== null ? videoScore.toFixed(1) + "/100" : "N/A (no video data)"}

    TASK:
    Evaluate the overall candidate considering technical knowledge, vocal delivery, AND answer quality/structure.
    Look for "Confidence Mismatches": If the technical score is high but vocal insights mention
    "vocal tremors" or "frequent hesitations," note that they may know the theory but lack
    confidence in explaining it under pressure.
    Also note delivery patterns: excessive filler words, hedging language, or lack of specificity.
    DIFFICULTY CURVE: If the candidate answered Easy/Medium questions well but struggled on Hard stretch questions, treat this as a POSITIVE signal for junior/mid roles — they demonstrated competence at their level and the system pushed them to their ceiling. Only treat Hard question struggles negatively for Senior roles where Hard is the expected baseline.
  `;

  const result = await generateStructured(prompt, FinalReportSchema);

  // 4b. Aggregate Video Score

  // 5. Final Combined Payload for the Database & UI
  const contentQuality =
    deliveryScore !== null
      ? technicalScore * 0.6 + deliveryScore * 0.4
      : technicalScore;

  let combined;
  if (voiceScore !== null && videoScore !== null) {
    combined = contentQuality * 0.41 + videoScore * 0.32 + voiceScore * 0.27;
  } else if (voiceScore !== null) {
    combined = contentQuality * 0.6 + voiceScore * 0.4;
  } else if (videoScore !== null) {
    combined = contentQuality * 0.56 + videoScore * 0.44;
  } else {
    combined = contentQuality;
  }

  const deliverySummary =
    turnsWithDelivery.length > 0
      ? {
          avgDeliveryScore: parseFloat(deliveryScore.toFixed(1)),
          totalFillers: turnsWithDelivery.reduce(
            (s, t) => s + t.deliveryAnalysis.fillerCount,
            0,
          ),
          totalHedging: turnsWithDelivery.reduce(
            (s, t) => s + t.deliveryAnalysis.hedgingCount,
            0,
          ),
          totalRestarts: turnsWithDelivery.reduce(
            (s, t) => s + t.deliveryAnalysis.sentenceRestarts,
            0,
          ),
          avgRelevance: parseFloat(
            (
              turnsWithDelivery.reduce(
                (s, t) => s + t.deliveryAnalysis.relevanceScore,
                0,
              ) / turnsWithDelivery.length
            ).toFixed(1),
          ),
          avgSpecificity: parseFloat(
            (
              turnsWithDelivery.reduce(
                (s, t) => s + t.deliveryAnalysis.specificityScore,
                0,
              ) / turnsWithDelivery.length
            ).toFixed(1),
          ),
        }
      : null;

  return {
    ...result,
    scores: {
      technical: parseFloat(technicalScore.toFixed(1)),
      voice: voiceScore !== null ? parseFloat(voiceScore.toFixed(1)) : null,
      video: videoScore !== null ? parseFloat(videoScore.toFixed(1)) : null,
      fusedScore,
      delivery:
        deliveryScore !== null ? parseFloat(deliveryScore.toFixed(1)) : null,
      contentQuality: parseFloat(contentQuality.toFixed(1)),
      combined: parseFloat(combined.toFixed(1)),
    },
    voiceSummary: hasVoiceData
      ? {
          overallLabel:
            voiceScore > 75
              ? "Highly Confident"
              : voiceScore > 50
                ? "Moderately Confident"
                : "Needs Improvement",
          allInsights: allVocalInsights,
          avgWPM: (
            voiceAnalyses.reduce((sum, v) => sum + v.wordsPerMinute, 0) /
            voiceAnalyses.length
          ).toFixed(0),
        }
      : null,
    deliverySummary,
  };
}
