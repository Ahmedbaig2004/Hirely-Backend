import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { z } from "zod";
import pdf from "pdf-parse/lib/pdf-parse.js";
import dotenv from "dotenv";
dotenv.config();

// 1. Initialize Gemini
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 0.2,
  apiKey: process.env.GOOGLE_API_KEY,
});

// 2. Define Output Structure
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
        reason: z.string().describe("Why this question was chosen"),
      })
    )
    .length(5)
    .describe("Exactly 5 interview questions"), // Matched with prompt below
});

// 3. Main Function
export async function generateInterviewContext(resumeBuffer, jobDescription) {
  console.log("📄 Parsing PDF Buffer...");

  const pdfData = await pdf(resumeBuffer);
  // Safety Truncate
  const resumeText = pdfData.text.substring(0, 15000);

  console.log("🤖 Generating Gap Analysis & Questions...");

  const structuredLlm = llm.withStructuredOutput(AnalysisSchema);

  const prompt = `
  You are an expert Technical Recruiter and Engineering Manager. Your task is to analyze the Candidate’s Resume in relation to the provided Job Description. Your primary goal is to identify any "missing skills: or "underrepresented qualifications" in the resume compared to the job description, but "also generate interview questions that cover the full scope" of the job description—ensuring that key areas of the role are assessed, not just the gaps.

JOB DESCRIPTION:
${jobDescription}

### CANDIDATE RESUME:
${resumeText}

TASK INSTRUCTIONS:

1. "Analyze the Entire Job Description":
   - Carefully compare the "entire Job Description" with the "Candidate Resume".
   - Identify all key "skills, qualifications, technologies, and experiences" required by the job.
   - Highlight "any missing skills" or "underrepresented qualifications" in the candidate's resume that are specifically outlined in the job description.
   - "Missing Skills": These are skills or qualifications mentioned in the job description but not found in the resume. This includes certifications, specific technical expertise, or experiences.
   - "Underrepresented Skills": These are skills that are briefly mentioned in the resume but lack the depth, frequency, or clarity expected by the job description.
   - For each missing or underrepresented skill, provide a "brief explanation" of why it is important for the role and the "potential impact" of its absence.

2. "Generate Exactly 5 Interview Questions":
   - "First Question": Begin with a "general introductory question": "Tell us about yourself with the person name got from the resume."
   
   - "Remaining 4 Questions": Generate "4 technical interview questions" based on the following guidelines:
     - "Cover the entire Job Description, not just the gaps.
     - use filler words like okay or great and many others to make it more human.
    -be like a human interviewer
    -can also ask about the proects from the resume
     - The questions should Test core competencies ,Assess real-world problem-solving,Cover multiple areas of the JD,Include soft skills if the JD mentions them
     - If the resume closely matches the job description, generate "advanced questions" that test the candidate's depth of knowledge and problem-solving skills in the core technical stack.
   
   - "For each question": Include the following:
     - "Question": A direct, specific question that tests the candidate’s proficiency.
     - "Topic": The technical area the question is targeting (e.g., Python, Data Structures, Cloud Computing, Project Management).
     - "Difficulty": Rate the difficulty as "easy", "medium", or "hard", based on the depth of knowledge required.
     - "Reason": Explain why this question was chosen, linking it to either a missing skill, underrepresented qualification, or key aspect of the job description.

3. "Additional Instructions":
   - If there are "soft skills" (e.g., communication, leadership) mentioned in the job description, generate questions to assess these as well. While the primary focus is technical, soft skills play a key role in many roles.
   - Ensure that questions reflect the "real-world challenges" the candidate will face on the job. Use practical, scenario-based questions to assess how the candidate applies their skills in practical situations.
   
4. "Formatting":
   - Provide the questions in a "clear, structured format", listing the question, topic, difficulty, and reason for each.
   - "Do not generate more than 6 questions"—keep the focus on the most important areas.
   - If a question relates to a gap or missing skill, explain "why this gap is critical" for the role.

  `;

  return await structuredLlm.invoke(prompt);
}
