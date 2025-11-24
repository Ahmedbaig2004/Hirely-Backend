import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { z } from "zod";
import pdf from "pdf-parse/lib/pdf-parse.js";
import dotenv from "dotenv";
dotenv.config();

// 1. Initialize Gemini
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 0.2, // Low temp for factual analysis
  apiKey: process.env.GOOGLE_API_KEY,
});

// 2. Define the Output Structure (The AI must follow this)
const AnalysisSchema = z.object({
  candidateSummary: z
    .string()
    .describe("A 2-sentence summary of the candidate"),
  gapAnalysis: z.object({
    matchScore: z.number().describe("0-100 score of how well they fit"),
    missingSkills: z
      .array(z.string())
      .describe("Skills in JD but missing in Resume"),
    feedback: z.string().describe("Constructive feedback on the gap"),
  }),
  questions: z
    .array(
      z.object({
        question: z.string(),
        topic: z.string(),
        difficulty: z.string(),
        reason: z
          .string()
          .describe(
            "Why this question was chosen (e.g. 'Tests missing skill: Redis')"
          ),
      })
    )
    .length(5)
    .describe(
      "Exactly 5 technical interview questions based on the gaps and requirements"
    ),
});

// 3. Main Function
export async function generateInterviewContext(resumeBuffer, jobDescription) {
  console.log("📄 Parsing PDF Buffer...");

  // A. Extract Text from PDF
  const pdfData = await pdf(resumeBuffer);
  // Truncate to ~15k chars to save tokens/money
  const resumeText = pdfData.text.substring(0, 15000);

  console.log("🤖 Generating Gap Analysis & Questions...");

  // B. Bind Schema to Model
  const structuredLlm = llm.withStructuredOutput(AnalysisSchema);

  // C. The Prompt
  const prompt = `
    You are an expert Technical Recruiter and Engineering Manager.
    
    Analyze the following Candidate Resume against the Job Description.
    
    JOB DESCRIPTION:
    ${jobDescription}

    CANDIDATE RESUME:
    ${resumeText}

    TASK:
    1. Identify the key gaps (what is the candidate missing?).
    2. Generate exactly 5 Technical Interview questions. 
       - Prioritize questions that test the "Missing Skills" to verify if they actually know them.
       - If the match is perfect, ask advanced questions about the core stack.
  `;

  // D. Execute
  return await structuredLlm.invoke(prompt);
}
