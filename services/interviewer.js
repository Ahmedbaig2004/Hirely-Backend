import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { z } from "zod";
import dotenv from "dotenv";
// 🆕 Note: The helper now returns both raw data AND generated insights
import { getVoiceAnalysesForInterview } from "./voiceAnalysisHelper.js";

dotenv.config();

// Updated to Gemini 3 Flash for peak performance
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-3-flash",
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
    voiceAnalyses = await getVoiceAnalysesForInterview(sessionId);
  } catch (e) {
    console.warn("⚠️ Voice analysis retrieval failed:", e.message);
  }

  // 3. Aggregate Communication Patterns
  const voiceScore =
    voiceAnalyses.length > 0
      ? voiceAnalyses.reduce((sum, v) => sum + v.confidenceLevel * 100, 0) /
        voiceAnalyses.length
      : 50;

  // Collect all unique bullet-point insights across the whole interview
  const allVocalInsights = [
    ...new Set(voiceAnalyses.flatMap((v) => v.insights)),
  ];

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
