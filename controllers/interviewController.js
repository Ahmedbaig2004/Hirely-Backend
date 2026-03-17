import { v4 as uuidv4 } from "uuid";
import fs from "fs";
import path from "path";
import axios from "axios";
import { generateInterviewContext } from "../services/analyzer.js";
import { stateManager } from "../services/stateManager.js";
import { transcribeAudio } from "../services/transcriber.js";
import {
  getNextQuestion,
  generateFinalReport,
} from "../services/interviewer.js";
import { evaluateAnswer } from "../services/retrieval.js";
import { prisma } from "../config/db.js";
import { generateAudio } from "../services/tts.js";
import { Redis } from "@upstash/redis";
import supabaseAdmin from "../config/supabaseAdmin.js";
import dotenv from "dotenv";

dotenv.config();

// Initialize Upstash Redis Client
const redisClient = new Redis({
  url: process.env.REDIS_URL,
  token: process.env.REDIS_TOKEN,
});

const MAX_QUESTIONS = 1;

/**
 * Initialize interview session
 */
export const initInterview = async (req, res) => {
  try {
    // ✅ Validate resume file
    if (!req.file) {
      return res.status(400).json({ error: "Resume file is required" });
    }

    if (req.file.size > 10 * 1024 * 1024) {
      // 10MB limit
      return res
        .status(400)
        .json({ error: "Resume file must be less than 10MB" });
    }

    // ✅ Validate job description
    const { userId, jobDescription } = req.body;

    if (!jobDescription || jobDescription.trim().length === 0) {
      return res.status(400).json({ error: "Job description is required" });
    }

    if (jobDescription.length > 10000) {
      return res
        .status(400)
        .json({ error: "Job description must be less than 10000 characters" });
    }

    if (!userId) {
      return res.status(400).json({ error: "User ID is required" });
    }

    const analysis = await generateInterviewContext(
      req.file.buffer,
      req.body.jobDescription,
    );

    const sessionId = uuidv4();
    const firstQuestion = analysis.questions[0];

    await stateManager.initSession(sessionId, {
      jobDescription: req.body.jobDescription,
      initialQuestions: analysis.questions,
      gapAnalysis: analysis.gapAnalysis,
      userId: userId || "anonymous",
    });

    await stateManager.updateCurrentQuestion(sessionId, firstQuestion);

    let audioBase64 = null;
    if (process.env.ENABLE_TTS === "true") {
      try {
        const audioBuffer = await generateAudio(firstQuestion.question);
        if (audioBuffer) audioBase64 = audioBuffer.toString("base64");
      } catch (e) {
        console.error("TTS Init Error:", e);
      }
    }

    res.json({
      sessionId,
      analysis,
      firstQuestion,
      audio: audioBase64,
    });
  } catch (e) {
    console.error("Init Error:", e);
    res.status(500).json({ error: "Init failed" });
  }
};

/**
 * Submit answer
 */
export const submitAnswer = async (req, res) => {
  try {
    const { sessionId, question } = req.body;
    let answerText = "";

    // 1. Transcription Logic
    if (req.file) {
      answerText = await transcribeAudio(req.file.buffer);
    } else if (req.body.answer) {
      answerText = req.body.answer;
    } else {
      return res.status(400).json({ error: "No audio provided." });
    }

    // 2. Textual Evaluation (RAG + Gemini)
    const evaluation = await evaluateAnswer(question, answerText);

    // 3. Save Turn to State (Redis)
    const updatedSession = await stateManager.saveTurn(
      sessionId,
      question,
      answerText,
      evaluation,
    );

    // 4. TRIGGER VOICE ANALYSIS (Background Task)
    if (req.file) {
      try {
        const turnIndex = updatedSession.history.length;
        const uploadDir = path.join(process.cwd(), "uploads", sessionId);
        const fileName = `turn_${turnIndex}.wav`;
        const audioPath = path.join(uploadDir, fileName);

        if (!fs.existsSync(uploadDir))
          fs.mkdirSync(uploadDir, { recursive: true });
        fs.writeFileSync(audioPath, req.file.buffer);

        axios
          .post(
            `${process.env.VOICE_SERVICE_URL || "http://localhost:8001"}/analyze-voice`,
            {
              turn_id: turnIndex,
              interview_id: sessionId,
              audio_path: audioPath,
              transcript: answerText,
            },
          )
          .catch((err) =>
            console.error("⚠️ Voice Service Trigger Failed:", err.message),
          );

        console.log(`🎤 Voice analysis triggered for turn ${turnIndex}`);
      } catch (voiceErr) {
        console.warn("⚠️ Voice Integration Error:", voiceErr.message);
      }
    }

    // 5. CHECK GAME OVER - Return immediately, let frontend poll + finalize
    if (updatedSession.history.length >= MAX_QUESTIONS) {
      console.log(
        "🏁 Interview Finished. Returning to frontend for voice analysis polling...",
      );
      return res.json({
        evaluation,
        isFinished: true,
        transcript: answerText,
      });
    }

    // 6. NEXT QUESTION & TTS LOGIC
    const queue = updatedSession.questionQueue;
    let nextQ =
      queue && queue.length > 0
        ? queue[0]
        : await getNextQuestion(updatedSession);
    await stateManager.updateCurrentQuestion(sessionId, nextQ);

    let audioBase64 = null;
    if (process.env.ENABLE_TTS === "true" && nextQ?.question) {
      try {
        const audioBuffer = await generateAudio(nextQ.question);
        if (audioBuffer) audioBase64 = audioBuffer.toString("base64");
      } catch (e) {
        console.error("TTS Error:", e);
      }
    }

    return res.json({
      evaluation,
      isFinished: false,
      nextQuestion: nextQ,
      transcript: answerText,
      audio: audioBase64,
    });
  } catch (error) {
    console.error("Submit Error:", error);
    res.status(500).json({ error: "Failed to process answer." });
  }
};

/**
 * Get voice analysis progress for a session
 */
export const getVoiceProgress = async (req, res) => {
  try {
    const { sessionId } = req.params;
    const total = MAX_QUESTIONS;
    let completed = 0;
    const statuses = [];

    for (let i = 1; i <= total; i++) {
      const data = await redisClient.get(`voice_analysis:${sessionId}:${i}`);
      if (data) {
        const parsed = typeof data === "string" ? JSON.parse(data) : data;
        if (parsed.status === "completed" || parsed.status === "failed") {
          completed++;
          statuses.push({ turn: i, status: parsed.status });
        } else {
          statuses.push({ turn: i, status: "processing" });
        }
      } else {
        statuses.push({ turn: i, status: "pending" });
      }
    }

    res.json({ completed, total, allDone: completed >= total, statuses });
  } catch (e) {
    console.error("Voice Progress Error:", e);
    res.status(500).json({ error: "Failed to check progress" });
  }
};

/**
 * Finalize interview - wait for voice analyses, generate report, persist to DB
 */
export const finalizeInterview = async (req, res) => {
  try {
    const { sessionId } = req.body;
    if (!sessionId) {
      return res.status(400).json({ error: "sessionId is required" });
    }

    // 1. Wait for all voice analyses to complete (poll Redis with retries)
    const MAX_RETRIES = 20;
    const RETRY_DELAY_MS = 1500;
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      let done = 0;
      for (let i = 1; i <= MAX_QUESTIONS; i++) {
        const data = await redisClient.get(`voice_analysis:${sessionId}:${i}`);
        if (data) {
          const parsed = typeof data === "string" ? JSON.parse(data) : data;
          if (parsed.status === "completed" || parsed.status === "failed")
            done++;
        }
      }
      if (done >= MAX_QUESTIONS) break;
      console.log(
        `⏳ Finalize: Voice analysis ${done}/${MAX_QUESTIONS} complete, waiting...`,
      );
      await new Promise((r) => setTimeout(r, RETRY_DELAY_MS));
    }

    // 2. Read session state from Redis
    const session = await stateManager.getSession(sessionId);
    if (!session) {
      return res.status(404).json({ error: "Session not found or expired" });
    }

    // 3. Enrich history with voice data from Redis
    const enrichedHistory = await Promise.all(
      session.history.map(async (turn, index) => {
        const turnNumber = index + 1;
        const rawVoiceData = await redisClient.get(
          `voice_analysis:${sessionId}:${turnNumber}`,
        );
        let voiceAnalysis = null;
        if (rawVoiceData) {
          voiceAnalysis =
            typeof rawVoiceData === "string"
              ? JSON.parse(rawVoiceData)
              : rawVoiceData;
        }
        return { ...turn, voiceAnalysis };
      }),
    );

    // 4. Generate the AI Final Report
    const finalReport = await generateFinalReport(
      sessionId,
      enrichedHistory,
      session.jobDescription,
      session.gapAnalysis,
    );

    const totalScore = enrichedHistory.reduce(
      (sum, turn) => sum + turn.score,
      0,
    );
    const averageScore = Math.round(totalScore / enrichedHistory.length);

    // 5. Upload audio files to Supabase Storage
    const audioUrls = {};
    const uploadDir = path.join(process.cwd(), "uploads", sessionId);
    console.log(
      `☁️  Uploading ${enrichedHistory.length} audio files to Supabase Storage...`,
    );
    for (let i = 1; i <= enrichedHistory.length; i++) {
      const localPath = path.join(uploadDir, `turn_${i}.wav`);
      if (fs.existsSync(localPath)) {
        try {
          const buffer = fs.readFileSync(localPath);
          const storagePath = `${session.userId}/${sessionId}/turn_${i}.webm`;
          const { error } = await supabaseAdmin.storage
            .from("interview-audios")
            .upload(storagePath, buffer, {
              contentType: "audio/webm",
              upsert: false,
            });
          if (!error) {
            const { data } = supabaseAdmin.storage
              .from("interview-audios")
              .getPublicUrl(storagePath);
            audioUrls[i] = data.publicUrl;
            console.log(`  ✅ Turn ${i} uploaded`);
          } else {
            console.warn(`  ⚠️ Turn ${i} upload failed:`, error.message);
          }
        } catch (uploadErr) {
          console.warn(`  ⚠️ Turn ${i} upload error:`, uploadErr.message);
        }
      }
    }

    // 6. Persist to PostgreSQL via Prisma
    await prisma.interview.create({
      data: {
        id: sessionId,
        userId: session.userId,
        jobDescription: session.jobDescription,
        finalScore: averageScore,
        finalFeedback: finalReport,
        turns: {
          create: enrichedHistory.map((turn, idx) => ({
            question: turn.question,
            answer: turn.answer,
            score: turn.score,
            feedback: turn.feedback,
            improvedAnswer: turn.betterAnswer,
            topic: turn.topic || "General",
            difficulty: turn.difficulty || "Medium",
            audioUrl: audioUrls[idx + 1] || null,
            voiceAnalysis: turn.voiceAnalysis
              ? {
                  create: {
                    confidenceLevel: turn.voiceAnalysis.confidenceLevel,
                    confidenceLabelText: turn.voiceAnalysis.confidenceLabelText,
                    speakingQuality: turn.voiceAnalysis.speakingQuality,
                    vocalStability: turn.voiceAnalysis.vocalStability,
                    speakingFluency: turn.voiceAnalysis.speakingFluency,
                    pitchMean: turn.voiceAnalysis.pitchMean,
                    pitchStd: turn.voiceAnalysis.pitchStd,
                    energyLevel: turn.voiceAnalysis.energyLevel,
                    wordsPerMinute: turn.voiceAnalysis.wordsPerMinute,
                    pauseRatio: turn.voiceAnalysis.pauseRatio,
                    jitter: turn.voiceAnalysis.jitter,
                    shimmer: turn.voiceAnalysis.shimmer,
                    allProbabilities: turn.voiceAnalysis.allProbabilities,
                    rawFeatures: turn.voiceAnalysis.rawFeatures,
                    status: "completed",
                    processedAt: new Date(turn.voiceAnalysis.processedAt),
                  },
                }
              : undefined,
          })),
        },
      },
    });

    // 7. Cleanup Redis + local audio files
    await stateManager.deleteSession(sessionId);
    for (let i = 1; i <= MAX_QUESTIONS; i++) {
      await redisClient.del(`voice_analysis:${sessionId}:${i}`);
    }
    if (fs.existsSync(uploadDir)) {
      fs.rmSync(uploadDir, { recursive: true, force: true });
      console.log(`🗑️  Cleaned up local audio files for ${sessionId}`);
    }

    console.log(`✅ Interview ${sessionId} finalized and persisted.`);
    return res.json({ success: true, finalReport });
  } catch (e) {
    console.error("Finalize Error:", e);
    res.status(500).json({ error: "Failed to finalize interview" });
  }
};
