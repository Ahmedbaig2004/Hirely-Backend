import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { z } from "zod";
import crypto from "crypto";
import pdf from "pdf-parse/lib/pdf-parse.js";
import { Redis } from "@upstash/redis";
import dotenv from "dotenv";
dotenv.config();

const redisClient = new Redis({
  url: process.env.REDIS_URL,
  token: process.env.REDIS_TOKEN,
});

function hashBuffer(buffer) {
  return crypto.createHash("sha256").update(buffer).digest("hex");
}

// 1. Initialize Gemini
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",
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
    .length(4)
    .describe("Exactly 4 interview questions"), // Matched with prompt below
});

// 3. Resume Validation
async function validateResumeContent(text) {
  const check = await llm.invoke(
    `You are a document classifier. Read the following text extracted from a PDF and determine if it is a professional resume or CV.

A resume typically contains: work experience, education, skills, contact info, job titles, or project descriptions.

If this is NOT a resume (e.g. a recipe, article, random text, story, manual, etc.), reply "no".
If this IS a resume/CV, reply "yes".

Reply with ONLY "yes" or "no".

TEXT:
${text.substring(0, 3000)}`
  );
  const answer = check.content.toString().trim().toLowerCase();
  return answer.startsWith("yes");
}

// 4. Main Function
export async function generateInterviewContext(resumeBuffer, jobDescription) {
  const bufferHash = hashBuffer(resumeBuffer);
  let resumeText;

  // --- Resume cache: skip PDF parse + validation if same file seen before ---
  try {
    const cached = await redisClient.get(`resume:${bufferHash}`);
    if (cached) {
      const parsed = typeof cached === "string" ? JSON.parse(cached) : cached;
      if (!parsed.isValid) {
        throw new Error("The uploaded document does not appear to be a resume. Please upload your professional resume or CV.");
      }
      resumeText = parsed.text;
      console.log("📄 Resume cache HIT — skipping parse + validation");
    }
  } catch (e) {
    // Re-throw validation errors, swallow Redis errors
    if (e.message?.includes("does not appear to be a resume")) throw e;
    console.warn("Resume cache read failed, proceeding without cache:", e.message);
  }

  if (!resumeText) {
    console.log("📄 Parsing PDF Buffer...");
    const pdfData = await pdf(resumeBuffer);
    resumeText = pdfData.text.substring(0, 15000);

    const isResume = await validateResumeContent(resumeText);

    // Cache result (valid or invalid) for 30 days
    try {
      await redisClient.set(
        `resume:${bufferHash}`,
        JSON.stringify({ text: resumeText, isValid: isResume }),
        { ex: 2592000 },
      );
    } catch (e) {
      console.warn("Resume cache write failed:", e.message);
    }

    if (!isResume) {
      throw new Error("The uploaded document does not appear to be a resume. Please upload your professional resume or CV.");
    }
    console.log("📄 Resume parsed + validated + cached");
  }

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
          - Donot explictly say that it is the final question on the final question or use finally


2. "Generate Exactly 4 Interview Questions":
   - "First Question": Begin with a "general introductory question": "Tell us about yourself with the person name got from the resume."
   
   - "Remaining 4 Questions": Generate "3 technical interview questions" based on the following guidelines:
      - Donot explictly say that it is the final question on the final question

     - "Cover the entire Job Description, not just the gaps.
    - Difficulty: Assign an explicit difficulty level ("Easy", "Medium", or "Hard") to each question. Follow this distribution based on the JD seniority:
      - JUNIOR/ENTRY/INTERN roles: Q1=Easy (intro), Q2=Easy, Q3=Medium, Q4=Medium. NEVER assign Hard to junior roles.
      - MID-LEVEL roles: Q1=Easy (intro), Q2=Medium, Q3=Medium, Q4=Hard.
      - SENIOR/LEAD/PRINCIPAL roles: Q1=Easy (intro), Q2=Medium, Q3=Hard, Q4=Hard.
    - **Tone:** Be conversational. Use natural filler words ("Okay", "Great", "Moving on").      -be like a human interviewer
      -Can also ask about the projects from the resume

     - The questions should Test core competencies ,Assess real-world problem-solving,Cover multiple areas of the JD,Include soft skills if the JD mentions them
     - If the resume closely matches the job description, generate "advanced questions" that test the candidate's depth of knowledge and problem-solving skills in the core technical stack.
3. "Additional Instructions":
       - Return the data strictly adhering to the JSON schema provided.
  `;

  return await structuredLlm.invoke(prompt);
}
