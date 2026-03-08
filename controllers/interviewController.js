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
import { prisma } from "../config/db.js"; // 👈 CHANGE 1: Use your central adapterimport { generateAudio } from "../services/tts.js";

const MAX_QUESTIONS = 9;

/**
 * Initialize interview session
 */
export const initInterview = async (req, res) => {
  try {
    if (!req.file) throw new Error("No resume uploaded");
    const { userId } = req.body;

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
          .post("http://localhost:8001/analyze-voice", {
            turn_id: turnIndex,
            interview_id: sessionId,
            audio_path: audioPath,
            transcript: answerText,
          })
          .catch((err) =>
            console.error("⚠️ Voice Service Trigger Failed:", err.message),
          );

        console.log(`🎤 Voice analysis triggered for turn ${turnIndex}`);
      } catch (voiceErr) {
        console.warn("⚠️ Voice Integration Error:", voiceErr.message);
      }
    }

    // 5. CHECK GAME OVER (Integrated Logic)
    if (updatedSession.history.length >= MAX_QUESTIONS) {
      console.log("🏁 Interview Finished. Syncing results...");

      // A. "Enrich" history by grabbing voice results from Redis
      const enrichedHistory = await Promise.all(
        updatedSession.history.map(async (turn, index) => {
          const turnNumber = index + 1;
          const redisKey = `voice_analysis:${sessionId}:${turnNumber}`;
          const rawVoiceData = await redisClient.get(redisKey);

          return {
            ...turn,
            voiceAnalysis: rawVoiceData ? JSON.parse(rawVoiceData) : null,
          };
        }),
      );

      // B. Generate the AI Final Report (Passing the enriched data)
      const finalReport = await generateFinalReport(
        sessionId,
        enrichedHistory,
        updatedSession.jobDescription,
        updatedSession.gapAnalysis,
      );

      const totalScore = enrichedHistory.reduce(
        (sum, turn) => sum + turn.score,
        0,
      );
      const averageScore = Math.round(totalScore / enrichedHistory.length);

      // C. Persist to PostgreSQL via Prisma
      await prisma.interview.create({
        data: {
          id: sessionId,
          userId: updatedSession.userId,
          jobDescription: updatedSession.jobDescription,
          finalScore: averageScore,
          finalFeedback: finalReport, // Integrated JSON report
          turns: {
            create: enrichedHistory.map((turn) => ({
              question: turn.question,
              answer: turn.answer,
              score: turn.score,
              feedback: turn.feedback,
              improvedAnswer: turn.betterAnswer,
              topic: turn.topic || "General",
              difficulty: turn.difficulty || "Medium",
              // Save the voice metadata if it exists
              voiceAnalysis: turn.voiceAnalysis
                ? {
                    create: {
                      confidenceLevel: turn.voiceAnalysis.confidenceLevel,
                      confidenceLabelText:
                        turn.voiceAnalysis.confidenceLabelText,
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

      // D. Final Cleanup
      await stateManager.deleteSession(sessionId);
      for (let i = 1; i <= MAX_QUESTIONS; i++) {
        await redisClient.del(`voice_analysis:${sessionId}:${i}`);
      }

      return res.json({
        evaluation,
        isFinished: true,
        finalReport,
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
