import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { z } from "zod";
import dotenv from "dotenv";
dotenv.config();

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
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
  summary: z.string().describe("3-sentence summary of performance"),
  strengths: z.array(z.string()),
  weaknesses: z.array(z.string()),
});

export async function getNextQuestion(session) {
  // 1. CHECK QUEUE (The Fixed Plan)
  // If there are still questions in the list (Q2, Q3...), return the next one.
  if (session.questionQueue && session.questionQueue.length > 0) {
    console.log(
      `📥 Queue has ${session.questionQueue.length} items. Serving next...`
    );
    return session.questionQueue[0];
  }

  // 2. ADAPTIVE MODE (The AI Brain)
  // If queue is empty, generate a new question based on history.
  console.log("🧠 Queue Empty! Generating ADAPTIVE Question...");

  const historyText = session.history
    .map((h) => `Q: ${h.question}\nA: ${h.answer}\nScore: ${h.score}/100`)
    .join("\n---\n");

  const prompt = `
    You are a Dynamic Technical Interviewer.
    
    JOB CONTEXT:
    ${session.jobDescription.substring(0, 500)}...

    INTERVIEW HISTORY:
    ${historyText}

    TASK:
    Generate the NEXT follow-up question.
    - If scores are low, ask easier fundamentals.
    - If scores are high, ask deeper system design.
    - Do NOT repeat questions.
  `;

  const structuredLlm = llm.withStructuredOutput(NextQuestionSchema);
  return await structuredLlm.invoke(prompt);
}
export async function generateFinalReport(history, jobDescription) {
  const structuredLlm = llm.withStructuredOutput(FinalReportSchema);

  const historyText = history
    .map((h) => `Q: ${h.question}\nA: ${h.answer}\nScore: ${h.score}/100`)
    .join("\n---\n");
  console.log("Generating Final Report with history:", historyText);

  const prompt = `
    You are a Hiring Manager making a final decision.
    
    JOB: ${jobDescription.substring(0, 500)}...
    
    FULL INTERVIEW TRANSCRIPT:
    ${historyText}

    CRITICAL INSTRUCTIONS:
    1. TRUST THE SCORES. A score of 0 or 10 means the candidate FAILED that question completely. It is NOT missing data.
    2. If the candidate consistently scores low (< 30), mark them as "No Hire".
    3. If the candidate gives repetitive or nonsense answers (e.g. "I have experience with React..." for every question), flag this as a "Weakness".


    TASK:
    Evaluate the candidate's overall performance.
    - Did they answer the core questions well?
    - Did they struggle with the adaptive "hard" questions?
    - Make a final hiring recommendation.
  `;

  return await structuredLlm.invoke(prompt);
}
