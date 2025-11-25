import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import multer from "multer";
import { v4 as uuidv4 } from "uuid";
import { generateInterviewContext } from "./services/analyzer.js";
import { stateManager } from "./services/stateManager.js";
import {
  getNextQuestion,
  generateFinalReport,
} from "./services/interviewer.js";
import { evaluateAnswer } from "./services/retrieval.js"; // Assuming you export this

dotenv.config();

const app = express();
const PORT = process.env.PORT || 4000;

// Middleware
app.use(cors());
app.use(express.json());

// Setup Multer for Resume Uploads (Stores file in RAM temporarily)
const upload = multer({ storage: multer.memoryStorage() });

// --- ROUTES ---

// 1. Health Check
app.get("/", (req, res) => {
  res.send("✅ HIRELY Brain is Active!");
});

// 2. Initialize Interview (Upload Resume -> Get Questions)
app.post("/api/init-interview", upload.single("resume"), async (req, res) => {
  try {
    // A. Call Analyzer (The Chef)
    const analysis = await generateInterviewContext(
      req.file.buffer,
      req.body.jobDescription
    );

    // B. Create Session ID
    const sessionId = uuidv4();

    // C. Save to Redis (The Fridge)
    await stateManager.initSession(sessionId, {
      jobDescription: req.body.jobDescription,
      initialQuestions: analysis.questions, // Stores the 6 questions
    });

    // D. Respond with Q1 immediately
    res.json({
      sessionId,
      analysis,
      firstQuestion: analysis.questions[0],
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Init failed" });
  }
});

// 2. SUBMIT ANSWER
const MAX_QUESTIONS = 10; // 6 Fixed + 4 Adaptive

app.post("/api/submit-answer", async (req, res) => {
  const { sessionId, question, answer } = req.body;

  // 1. Grade & Save (Same as before)
  const evaluation = await evaluateAnswer(question, answer);
  const updatedSession = await stateManager.saveTurn(
    sessionId,
    0,
    question,
    answer,
    evaluation
  );

  // 2. CHECK FOR END OF INTERVIEW
  // If we have asked enough questions, STOP.
  if (updatedSession.history.length >= MAX_QUESTIONS) {
    console.log("🏁 Interview Finished. Generating Report...");

    const finalReport = await generateFinalReport(
      updatedSession.history,
      updatedSession.jobDescription
    );

    // Send "isFinished: true" so frontend knows to show the Result Page
    return res.json({
      evaluation,
      isFinished: true,
      finalReport,
    });
  }

  // 3. Continue if not finished (Same logic as before)
  const queue = updatedSession.questionQueue;
  let nextQ;

  if (queue && queue.length > 0) {
    nextQ = queue[0];
  } else {
    nextQ = await getNextQuestion(updatedSession);
  }

  res.json({
    evaluation,
    isFinished: false,
    nextQuestion: nextQ,
  });
});

// 3. Evaluate Answer (RAG Grading)
app.post("/api/evaluate", async (req, res) => {
  try {
    const { question, answer } = req.body;
    console.log(`⚖️ Grading: ${question}`);

    // This calls your existing RAG logic
    const result = await evaluateAnswer(question, answer);

    res.json(result);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "Failed to evaluate answer" });
  }
});

// Start the Server
app.listen(PORT, () => {
  console.log(`\n🚀 HIRELY Backend running on http://localhost:${PORT}`);
});
