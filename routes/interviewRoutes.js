import express from "express";
import multer from "multer";
import rateLimit from "express-rate-limit";
import {
  initInterview,
  submitAnswer,
  getVoiceProgress,
  finalizeInterview,
  getFinalizeStatus,
  cancelInterview,
} from "../controllers/interviewController.js";

const router = express.Router();

// Multer configuration for file uploads
const upload = multer({ storage: multer.memoryStorage() });

// AI LIMITER (Cost Control)
// Limit expensive AI calls per 1 hour
const aiLimitMax = Number(process.env.AI_RATE_LIMIT_MAX ?? 1000);
const aiLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,
  max: aiLimitMax,
  message: "AI Processing Limit Reached. Please wait.",
});

// Initialize interview (Protected by AI Limiter)
router.post(
  "/init-interview",
  aiLimiter,
  upload.single("resume"),
  initInterview,
);

// Submit answer (Protected by AI Limiter)
router.post(
  "/submit-answer",
  aiLimiter,
  upload.fields([
    { name: "audio", maxCount: 1 },
    { name: "video", maxCount: 1 },
  ]),
  submitAnswer,
);

// Voice analysis progress polling
router.get("/voice-progress/:sessionId", getVoiceProgress);

// Finalize interview (kicks off background processing)
router.post("/finalize-interview", finalizeInterview);

// Poll finalization status
router.get("/finalize-status/:sessionId", getFinalizeStatus);

// Cancel active interview — deletes Redis session
router.delete("/:sessionId", cancelInterview);

export default router;
