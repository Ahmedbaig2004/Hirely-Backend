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
import { analyzeDelivery } from "../services/deliveryAnalyzer.js";
import { prisma } from "../config/db.js";
import { generateAudio } from "../services/tts.js";
import { translateIfNeeded, isRomanUrdu } from "../services/translator.js";
import { Redis } from "@upstash/redis";
import supabaseAdmin from "../config/supabaseAdmin.js";
import dotenv from "dotenv";

dotenv.config();

// Initialize Upstash Redis Client
const redisClient = new Redis({
  url: process.env.REDIS_URL,
  token: process.env.REDIS_TOKEN,
});

const MAX_QUESTIONS = 2;

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
    if (e.message?.includes("does not appear to be a resume")) {
      return res.status(400).json({ error: e.message });
    }
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

    // 1. Transcription Logic + determine answer mode
    let answerMode = "chat";
    let detectedLanguage = "en";
    if (req.file) {
      const { transcript, language } = await transcribeAudio(req.file.buffer);
      answerText = transcript;
      detectedLanguage = language;
      answerMode = "audio";
    } else if (req.body.answer) {
      answerText = req.body.answer;
      detectedLanguage = isRomanUrdu(answerText) ? "ur" : "en";
    } else {
      return res.status(400).json({ error: "No answer provided." });
    }

    // 2. Fetch session for job description + current question difficulty
    const session = await stateManager.getSession(sessionId);
    const jobDescription = session?.jobDescription || "";
    const currentDifficulty = session?.currentQuestion?.difficulty || "Medium";

    // 2.5. Translate Roman Urdu → English for evaluation (Urdu answers only)
    // English answers are used directly; original Roman Urdu is kept for display/storage
    let evaluationText = answerText;
    if (detectedLanguage === "ur") {
      const { translatedText } = await translateIfNeeded(answerText);
      evaluationText = translatedText;
    }

    // 3. Generate Next Question only — evaluation and delivery analysis are deferred to
    // finalization where all turns are scored in parallel (no per-turn LLM grading here).
    // getNextQuestion reads the raw transcript directly so no score is needed.
    // Trigger parallel generation when queue has ≤1 item: after saveTurn shifts the queue,
    // it will be empty, and parallelNextQ will be ready instead of blocking on a fallback call.
    const nextQuestionNeeded =
      !session.questionQueue || session.questionQueue.length <= 1;
    const parallelNextQ = nextQuestionNeeded
      ? await getNextQuestion(session, { question, answer: answerText })
      : null;

    // 4. Save Turn to State (Redis) — score/feedback null until finalization
    const updatedSession = await stateManager.saveTurn(
      sessionId,
      question,
      answerText,
      null,        // evaluation deferred
      answerMode,
      null,        // deliveryAnalysis deferred
      evaluationText,
      detectedLanguage,
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
        isFinished: true,
        transcript: answerText,
      });
    }

    // 6. NEXT QUESTION & TTS LOGIC
    // Use pre-generated queue first, then the parallel-generated question.
    const queue = updatedSession.questionQueue;
    let nextQ =
      queue && queue.length > 0
        ? queue[0]
        : parallelNextQ ?? (await getNextQuestion(updatedSession));
    // Pass updatedSession so updateCurrentQuestion skips the redundant Redis GET
    await stateManager.updateCurrentQuestion(sessionId, nextQ, updatedSession);

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

    // Get session to count only audio turns
    const session = await stateManager.getSession(sessionId);
    const history = session?.history || [];
    const audioTurnIndices = [];
    history.forEach((turn, idx) => {
      // Include audio turns and legacy turns (no answerMode field)
      if (turn.answerMode === "audio" || !turn.answerMode) {
        audioTurnIndices.push(idx + 1);
      }
    });

    // If no audio turns, everything is done
    const total = audioTurnIndices.length;
    if (total === 0) {
      return res.json({ completed: 0, total: 0, allDone: true, statuses: [] });
    }

    let completed = 0;
    const statuses = [];

    for (const turnIdx of audioTurnIndices) {
      const data = await redisClient.get(
        `voice_analysis:${sessionId}:${turnIdx}`,
      );
      if (data) {
        const parsed = typeof data === "string" ? JSON.parse(data) : data;
        if (parsed.status === "completed" || parsed.status === "failed") {
          completed++;
          statuses.push({ turn: turnIdx, status: parsed.status });
        } else {
          statuses.push({ turn: turnIdx, status: "processing" });
        }
      } else {
        statuses.push({ turn: turnIdx, status: "pending" });
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
  const { sessionId } = req.body;
  if (!sessionId) {
    return res.status(400).json({ error: "sessionId is required" });
  }
  const uploadDir = path.join(process.cwd(), "uploads", sessionId);

  try {

    // 1. Read session state from Redis
    const session = await stateManager.getSession(sessionId);
    if (!session) {
      return res.status(404).json({ error: "Session not found or expired" });
    }

    // 2. Wait for voice analyses to complete (only for audio turns)
    // All keys checked in parallel per attempt; max wait reduced to 10s.
    const audioTurnIndices = session.history
      .map((turn, idx) => (turn.answerMode === "audio" ? idx + 1 : null))
      .filter((idx) => idx !== null);

    if (audioTurnIndices.length > 0) {
      const MAX_RETRIES = 10;
      const RETRY_DELAY_MS = 1000;
      for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        const checks = await Promise.all(
          audioTurnIndices.map((turnIdx) =>
            redisClient.get(`voice_analysis:${sessionId}:${turnIdx}`),
          ),
        );
        const done = checks.filter((data) => {
          if (!data) return false;
          const parsed = typeof data === "string" ? JSON.parse(data) : data;
          return parsed.status === "completed" || parsed.status === "failed";
        }).length;
        if (done >= audioTurnIndices.length) break;
        console.log(
          `⏳ Finalize: Voice analysis ${done}/${audioTurnIndices.length} complete, waiting...`,
        );
        await new Promise((r) => setTimeout(r, RETRY_DELAY_MS));
      }
    }

    // 3. Evaluate all turns in parallel (deferred from interview hot path)
    // Each turn stored evaluationText (translated if Urdu) and detectedLanguage for this step.
    console.log(`⚖️  Evaluating ${session.history.length} turns in parallel...`);
    const [evaluations, deliveryAnalyses] = await Promise.all([
      Promise.all(
        session.history.map((turn) =>
          evaluateAnswer(
            turn.question,
            turn.evaluationText || turn.answer,
            session.jobDescription,
            turn.difficulty || "Medium",
          ),
        ),
      ),
      Promise.all(
        session.history.map((turn) =>
          analyzeDelivery(
            turn.answer,
            turn.question,
            turn.detectedLanguage || "en",
          ),
        ),
      ),
    ]);

    // Merge evaluation results back into history
    const scoredHistory = session.history.map((turn, idx) => ({
      ...turn,
      score: evaluations[idx]?.score ?? 0,
      feedback: evaluations[idx]?.feedback ?? "",
      betterAnswer: evaluations[idx]?.betterAnswer ?? null,
      deliveryAnalysis: deliveryAnalyses[idx] ?? null,
    }));

    // 4. Enrich scored history with voice data from Redis (batch fetch — one roundtrip)
    const voiceKeys = scoredHistory.map((_, index) =>
      `voice_analysis:${sessionId}:${index + 1}`,
    );
    const voiceDataRaw = await Promise.all(
      voiceKeys.map((key) => redisClient.get(key)),
    );
    const enrichedHistory = scoredHistory.map((turn, index) => {
      const raw = voiceDataRaw[index];
      const voiceAnalysis = raw
        ? (typeof raw === "string" ? JSON.parse(raw) : raw)
        : null;
      return { ...turn, voiceAnalysis };
    });

    // Pre-extract non-null voice records for generateFinalReport to avoid a duplicate Redis read
    const prefetchedVoiceData = enrichedHistory
      .map((turn) => turn.voiceAnalysis)
      .filter((v) => v !== null);

    const totalScore = enrichedHistory.reduce(
      (sum, turn) => sum + (turn.score ?? 0),
      0,
    );
    const averageScore = Math.round(totalScore / enrichedHistory.length);

    // 5+6. Generate AI report and upload audio files in parallel (no dependency between them)
    console.log(
      `☁️  Uploading ${enrichedHistory.length} audio files to Supabase Storage...`,
    );
    const [finalReport, audioUrls] = await Promise.all([
      generateFinalReport(
        sessionId,
        enrichedHistory,
        session.jobDescription,
        session.gapAnalysis,
        prefetchedVoiceData,
      ),
      (async () => {
        const urls = {};
        await Promise.all(
          Array.from({ length: enrichedHistory.length }, (_, i) => i + 1).map(
            async (i) => {
              const localPath = path.join(uploadDir, `turn_${i}.wav`);
              if (!fs.existsSync(localPath)) return;
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
                  urls[i] = data.publicUrl;
                  console.log(`  ✅ Turn ${i} uploaded`);
                } else {
                  console.warn(`  ⚠️ Turn ${i} upload failed:`, error.message);
                }
              } catch (uploadErr) {
                console.warn(
                  `  ⚠️ Turn ${i} upload error:`,
                  uploadErr.message,
                );
              }
            },
          ),
        );
        return urls;
      })(),
    ]);

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
            score: turn.score ?? 0,
            feedback: turn.feedback || "",
            improvedAnswer: turn.betterAnswer,
            topic: turn.topic || "General",
            difficulty: turn.difficulty || "Medium",
            answerMode: turn.answerMode || null,
            audioUrl: audioUrls[idx + 1] || null,
            deliveryScore: turn.deliveryAnalysis?.deliveryScore ?? null,
            fillerCount: turn.deliveryAnalysis?.fillerCount ?? null,
            hedgingCount: turn.deliveryAnalysis?.hedgingCount ?? null,
            sentenceRestarts: turn.deliveryAnalysis?.sentenceRestarts ?? null,
            relevanceScore: turn.deliveryAnalysis?.relevanceScore ?? null,
            specificityScore: turn.deliveryAnalysis?.specificityScore ?? null,
            deliveryFeedback: turn.deliveryAnalysis ?? null,
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

    console.log(`✅ Interview ${sessionId} finalized and persisted.`);
    return res.json({ success: true, finalReport });
  } catch (e) {
    console.error("Finalize Error:", e);
    res.status(500).json({ error: "Failed to finalize interview" });
  } finally {
    // 7. Always cleanup Redis + local audio files, even on error
    try {
      await stateManager.deleteSession(sessionId);
      for (let i = 1; i <= MAX_QUESTIONS; i++) {
        await redisClient.del(`voice_analysis:${sessionId}:${i}`);
      }
      if (fs.existsSync(uploadDir)) {
        fs.rmSync(uploadDir, { recursive: true, force: true });
        console.log(`🗑️  Cleaned up local audio files for ${sessionId}`);
      }
    } catch (cleanupErr) {
      console.warn("⚠️ Cleanup error:", cleanupErr.message);
    }
  }
};
