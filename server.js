import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import multer from "multer";
import { v4 as uuidv4 } from "uuid";
import fs from "fs";
import path from "path";
import rateLimit from "express-rate-limit";
import { fileURLToPath } from "url";
import { generateInterviewContext } from "./services/analyzer.js";
import { stateManager } from "./services/stateManager.js";
import { transcribeAudio } from "./services/transcriber.js";
import {
  getNextQuestion,
  generateFinalReport,
} from "./services/interviewer.js";
import { evaluateAnswer } from "./services/retrieval.js";
import { PrismaClient } from "./generated/prisma/index.js";

dotenv.config();
const prisma = new PrismaClient();
const app = express();
const PORT = process.env.PORT || 4000;

// --- 1. BOILERPLATE FOR PATHS (Essential for ES Modules) ---
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- 2. ROBUST CORS (Fixes the "Empty Headers" issue) ---
app.use(
  cors({
    origin: true, // Allow any origin (127.0.0.1, localhost, etc.)
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
    credentials: true,
  })
);
const safeDelete = (path) => {
  if (!path) return;
  // Wait 1 second to let Windows release the lock
  setTimeout(() => {
    try {
      if (fs.existsSync(path)) {
        fs.unlinkSync(path);
        console.log(`🧹 Cleanup success: ${path}`);
      }
    } catch (err) {
      console.warn(
        `⚠️ Cleanup warning: Could not delete ${path} - ${err.message}`
      );
    }
  }, 1000);
};

// --- 3. CREATE UPLOAD FOLDER (Prevents Crash) ---
const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) {
  console.log(`📂 Creating upload folder at: ${uploadDir}`);
  fs.mkdirSync(uploadDir, { recursive: true });
}

// --- 4. MIDDLEWARE ---
app.use(express.json());

// Limiter
const aiLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,
  max: 100,
  message: "AI Processing Limit Reached.",
});

// Multer Config
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir); // Uses the folder we created above
  },
  filename: (req, file, cb) => {
    // Sanitizes filename to prevent weird character errors
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    const safeName = file.originalname.replace(/[^a-zA-Z0-9.]/g, "_");
    cb(null, uniqueSuffix + "-" + safeName);
  },
});
const upload = multer({ storage: storage });

// --- 5. ROUTES ---

app.get("/", (req, res) => {
  res.send("✅ HIRELY Brain is Active!");
});

// Initialize Interview
app.post("/api/init-interview", upload.single("resume"), async (req, res) => {
  const filePath = req.file?.path;
  console.log("📥 Received Init Request. File:", filePath);

  try {
    if (!filePath) throw new Error("No resume uploaded");

    const fileBuffer = fs.readFileSync(filePath);

    // Process logic
    const analysis = await generateInterviewContext(
      fileBuffer,
      req.body.jobDescription
    );
    const sessionId = uuidv4();
    await stateManager.initSession(sessionId, {
      jobDescription: req.body.jobDescription,
      initialQuestions: analysis.questions,
    });

    console.log("✅ Session Created, Sending Response...");

    // Send Response
    res.json({
      sessionId,
      analysis,
      firstQuestion: analysis.questions[0],
    });
  } catch (e) {
    console.error("❌ INIT ERROR:", e.message);
    res.status(500).json({ error: e.message });
  } finally {
    // 3. SAFE CLEANUP (Using the helper)
    // This runs completely separate from the response logic
    safeDelete(filePath);
  }
});

// Submit Answer
app.post(
  "/api/submit-answer",
  aiLimiter,
  upload.single("audio"),
  async (req, res) => {
    const filePath = req.file?.path;

    try {
      const { sessionId, question } = req.body;
      let answerText = "";

      if (filePath) {
        console.log(`🎧 Processing Audio Answer...`);
        const fileBuffer = fs.readFileSync(filePath);
        answerText = await transcribeAudio(fileBuffer);
      } else if (req.body.answer) {
        answerText = req.body.answer;
      } else {
        return res.status(400).json({ error: "No answer provided." });
      }

      const evaluation = await evaluateAnswer(question, answerText);
      const updatedSession = await stateManager.saveTurn(
        sessionId,
        question,
        answerText,
        evaluation
      );

      // Check if finished
      const MAX_QUESTIONS = 6; // Set to 6 for testing, 10 for production
      if (updatedSession.history.length >= MAX_QUESTIONS) {
        const finalReport = await generateFinalReport(
          updatedSession.history,
          updatedSession.jobDescription
        );

        await prisma.interview.create({
          data: {
            id: sessionId,
            jobDescription: updatedSession.jobDescription,
            finalScore: 0,
            finalFeedback: finalReport,
            turns: {
              create: updatedSession.history.map((t) => ({
                question: t.question,
                answer: t.answer,
                score: t.score,
                feedback: t.feedback,
                improvedAnswer: t.betterAnswer,
              })),
            },
          },
        });

        await stateManager.deleteSession(sessionId);
        return res.json({
          evaluation,
          isFinished: true,
          finalReport,
          transcript: answerText,
        });
      }

      // Next Question
      const queue = updatedSession.questionQueue;
      const nextQ =
        queue && queue.length > 0
          ? queue[0]
          : await getNextQuestion(updatedSession);

      res.json({
        evaluation,
        isFinished: false,
        nextQuestion: nextQ,
        transcript: answerText,
      });
    } catch (error) {
      console.error("❌ SUBMIT ERROR:", error);
      res.status(500).json({ error: "Failed to process answer." });
    } finally {
      if (filePath && fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
    }
  }
);

app.post("/api/evaluate", async (req, res) => {
  try {
    const { question, answer } = req.body;
    const result = await evaluateAnswer(question, answer);
    res.json(result);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Eval failed" });
  }
});

app.listen(PORT, () => {
  console.log(`\n🚀 HIRELY Backend running on http://localhost:${PORT}`);
  console.log(`📂 Uploads will be saved to: ${uploadDir}`);
});
