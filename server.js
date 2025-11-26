import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import multer from "multer";
import { v4 as uuidv4 } from "uuid";
import { generateInterviewContext } from "./services/analyzer.js";
import { stateManager } from "./services/stateManager.js";
import { transcribeAudio } from "./services/transcriber.js";

import {
  getNextQuestion,
  generateFinalReport,
} from "./services/interviewer.js";
import { evaluateAnswer } from "./services/retrieval.js"; // Assuming you export this
import { PrismaClient } from "./generated/prisma/index.js";
const prisma = new PrismaClient();
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

app.post("/api/submit-answer", upload.single("audio"), async (req, res) => {
  try {
    const { sessionId, question } = req.body;
    let answerText = "";

    // 🎤 STEP 1: Determine Input Type (Audio vs Text)
    if (req.file) {
      console.log(`🎧 Processing Audio Answer (${req.file.size} bytes)...`);
      // Convert Audio -> Text using your local Whisper model
      answerText = await transcribeAudio(req.file.buffer);
    } else if (req.body.answer) {
      // Fallback to standard text input
      answerText = req.body.answer;
    } else {
      return res.status(400).json({ error: "No audio or text provided." });
    }

    console.log(`🗣️  Final Answer: "${answerText}"`);

    // 📝 STEP 2: Grade & Save (Using answerText)
    const evaluation = await evaluateAnswer(question, answerText);

    const updatedSession = await stateManager.saveTurn(
      sessionId,
      question,
      answerText, // <--- Pass the transcribed text here
      evaluation
    );

    // 🏁 STEP 3: Check for "Game Over"
    if (updatedSession.history.length >= MAX_QUESTIONS) {
      console.log("🏁 Interview Finished. Generating Report...");

      const finalReport = await generateFinalReport(
        updatedSession.history,
        updatedSession.jobDescription
      );
      console.log("💾 Persisting to Database...");
      await prisma.interview.create({
        data: {
          id: sessionId, // Use the same ID
          jobDescription: updatedSession.jobDescription,
          finalScore: 0, // You can calculate average score here if you want
          finalFeedback: finalReport,
          turns: {
            create: updatedSession.history.map((turn) => ({
              question: turn.question,
              answer: turn.answer,
              score: turn.score,
              feedback: turn.feedback,
              improvedAnswer: turn.betterAnswer,
            })),
          },
        },
      });

      // 3. DELETE FROM REDIS (Cleanup)
      console.log("🧹 Cleaning Redis...");
      // We don't need the cache anymore because it's in Postgres
      // Access redis client via stateManager (you might need to export client or add a delete method)
      // For now, let's assume stateManager has a deleteSession method:
      await stateManager.deleteSession(sessionId);

      return res.json({
        evaluation,
        isFinished: true,
        finalReport,
        transcript: answerText, // Return this so the UI can show what the AI heard
      });
    }

    // ➡️ STEP 4: Get Next Question (Queue vs Adaptive)
    const queue = updatedSession.questionQueue;
    let nextQ;

    if (queue && queue.length > 0) {
      // Still creating the baseline
      nextQ = queue[0];
    } else {
      // Baseline done, drill down adaptively
      nextQ = await getNextQuestion(updatedSession);
    }

    return res.json({
      evaluation,
      isFinished: false,
      nextQuestion: nextQ,
      transcript: answerText, // Return transcript for UI feedback
    });
  } catch (error) {
    console.error("❌ Answer Submission Error:", error);
    res.status(500).json({ error: "Failed to process answer." });
  }
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
