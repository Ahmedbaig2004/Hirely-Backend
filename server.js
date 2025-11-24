import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import multer from "multer";
import { evaluateAnswer } from "./services/retrieval.js"; // Assuming you export this
import { generateInterviewContext } from "./services/analyzer.js"; // We will create this next

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
    if (!req.file) return res.status(400).json({ error: "No resume uploaded" });
    if (!req.body.jobDescription)
      return res.status(400).json({ error: "No Job Description" });

    console.log("📄 Processing Resume...");
    const analysis = await generateInterviewContext(
      req.file.buffer,
      req.body.jobDescription
    );

    res.json(analysis);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "Failed to analyze resume" });
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
