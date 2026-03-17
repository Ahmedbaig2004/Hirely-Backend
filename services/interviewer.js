import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { z } from "zod";
import dotenv from "dotenv";
// 🆕 Note: The helper now returns both raw data AND generated insights
import { getVoiceAnalysesForInterview } from "./voiceAnalysisHelper.js";

dotenv.config();

// Updated to Gemini 3 Flash for peak performance
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",
  apiKey: process.env.GOOGLE_API_KEY,
});

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

  // Build per-turn summaries. v2.0+ sessions include SHAP-identified drivers;
  // older sessions fall back to raw metrics. Both paths produce useful Gemini input.
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
      // v2.0: include top SHAP drivers so Gemini knows what the model actually detected
      return {
        ...base,
        topFactors: shapExplanations.slice(0, 3).map((e) => ({
          feature: e.label,
          impact: e.direction,        // "increased" or "decreased"
          explanation: e.explanation, // already-written coaching tip
        })),
      };
    }

    // Fallback for pre-v2.0 sessions
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
- Write in second person ("You maintained steady projection throughout")
- Keep each observation to one sentence
- Do not refer to specific turn numbers`;

  try {
    const structuredLlm = llm.withStructuredOutput(VoiceInsightsSchema);
    const result = await structuredLlm.invoke(prompt);
    return result.insights;
  } catch (e) {
    console.warn("⚠️ AI voice insights generation failed:", e.message);
    return [];
  }
}

/**
 * Adaptive Questioning Logic
 */
export async function getNextQuestion(session) {
  if (session.questionQueue && session.questionQueue.length > 0) {
    return session.questionQueue[0];
  }

  const historyText = session.history
    .map((h) => `Q: ${h.question}\nA: ${h.answer}\nScore: ${h.score}/100`)
    .join("\n---\n");

  const prompt = `
    You are a Dynamic Technical Interviewer.
    JOB CONTEXT: ${session.jobDescription.substring(0, 500)}...
    INTERVIEW HISTORY: ${historyText}

    TASK: Generate the NEXT follow-up question. Adapt based on previous answers.
  `;

  const structuredLlm = llm.withStructuredOutput(NextQuestionSchema);
  return await structuredLlm.invoke(prompt);
}

/**
 * Final Report Generation with Integrated Voice Insights
 */
export async function generateFinalReport(
  sessionId,
  history,
  jobDescription,
  gapAnalysis,
) {
  // 1. Calculate Technical Score
  const technicalScore =
    history.length > 0
      ? history.reduce((sum, turn) => sum + turn.score, 0) / history.length
      : 0;

  // 2. Fetch interpreted Voice Data
  let voiceAnalyses = [];
  try {
    console.log(`⏳ Extracting vocal insights for: ${sessionId}`);
    // This now returns objects containing .insights and .metrics thanks to your helper update
    voiceAnalyses = await getVoiceAnalysesForInterview(
      sessionId,
      history.length,
    );
  } catch (e) {
    console.warn("⚠️ Voice analysis retrieval failed:", e.message);
  }

  // 3. Aggregate Communication Patterns
  const voiceScore =
    voiceAnalyses.length > 0
      ? voiceAnalyses.reduce((sum, v) => sum + v.confidenceLevel * 100, 0) /
        voiceAnalyses.length
      : 50;

  // AI-generated insights: one LLM call synthesizes all turns into coherent, non-repetitive feedback
  console.log(`🤖 Generating AI voice insights for ${voiceAnalyses.length} turns...`);
  const allVocalInsights = await generateAIVoiceInsights(voiceAnalyses);

  const historyText = history
    .map((h, i) => `Q${i + 1}: ${h.question}\nTechnical Score: ${h.score}/100`)
    .join("\n\n");

  // 4. The "Expert Recruiter" Prompt
  const prompt = `
    You are an expert Technical Hiring Manager. 

    TECHNICAL DATA:
    - Score: ${technicalScore.toFixed(1)}/100
    - History: ${historyText}
    - Resume Gaps: ${JSON.stringify(gapAnalysis)}

    COMMUNICATION DATA:
    - Confidence Score: ${voiceScore.toFixed(1)}/100
    - Observations: ${allVocalInsights.length > 0 ? allVocalInsights.join("; ") : "Stable and confident delivery."}

    TASK:
    Evaluate the overall candidate. 
    Look for "Confidence Mismatches": If the technical score is high but vocal insights mention 
    "vocal tremors" or "frequent hesitations," note that they may know the theory but lack 
    confidence in explaining it under pressure.
  `;

  const structuredLlm = llm.withStructuredOutput(FinalReportSchema);
  const result = await structuredLlm.invoke(prompt);

  // 5. Final Combined Payload for the Database & UI
  return {
    ...result,
    scores: {
      technical: parseFloat(technicalScore.toFixed(1)),
      voice: parseFloat(voiceScore.toFixed(1)),
      combined: parseFloat(
        (technicalScore * 0.6 + voiceScore * 0.4).toFixed(1),
      ),
    },
    voiceSummary: {
      overallLabel:
        voiceScore > 75
          ? "Highly Confident"
          : voiceScore > 50
            ? "Moderately Confident"
            : "Needs Improvement",
      allInsights: allVocalInsights,
      avgWPM:
        voiceAnalyses.length > 0
          ? (
              voiceAnalyses.reduce((sum, v) => sum + v.wordsPerMinute, 0) /
              voiceAnalyses.length
            ).toFixed(0)
          : "N/A",
    },
  };
}
