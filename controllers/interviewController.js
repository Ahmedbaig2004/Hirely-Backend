import { v4 as uuidv4 } from "uuid";
import { generateInterviewContext } from "../services/analyzer.js";
import { stateManager } from "../services/stateManager.js";
import { transcribeAudio } from "../services/transcriber.js";
import {
  getNextQuestion,
  generateFinalReport,
} from "../services/interviewer.js";
import { evaluateAnswer } from "../services/retrieval.js";
import { PrismaClient } from "../generated/prisma/index.js";
import { generateAudio } from "../services/tts.js"; // Deepgram Service

const prisma = new PrismaClient();
const MAX_QUESTIONS = 10;

/**
 * Initialize interview session
 */
export const initInterview = async (req, res) => {
  try {
    if (!req.file) throw new Error("No resume uploaded");

    const analysis = await generateInterviewContext(
      req.file.buffer,
      req.body.jobDescription
    );

    const sessionId = uuidv4();
    const firstQuestion = analysis.questions[0];

    await stateManager.initSession(sessionId, {
      jobDescription: req.body.jobDescription,
      initialQuestions: analysis.questions,
      gapAnalysis: analysis.gapAnalysis,
    });

    await stateManager.updateCurrentQuestion(sessionId, firstQuestion);

    // ✅ GENERATE AUDIO FOR QUESTION 1
    // This ensures the very first question is spoken too.
    let audioBase64 = null;
    if (process.env.ENABLE_TTS === "true") {
      try {
        console.time("TTS Init");
        const audioBuffer = await generateAudio(firstQuestion.question);
        console.timeEnd("TTS Init");
        if (audioBuffer) audioBase64 = audioBuffer.toString("base64");
      } catch (e) {
        console.error("TTS Init Error:", e);
      }
    } else {
      console.log("🔕 TTS Disabled (Saving Credits)");
    }

    res.json({
      sessionId,
      analysis,
      firstQuestion,
      audio: audioBase64, // <--- Send Q1 Audio
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

    if (req.file) {
      answerText = await transcribeAudio(req.file.buffer);
    } else if (req.body.answer) {
      answerText = req.body.answer;
    } else {
      return res.status(400).json({ error: "No audio provided." });
    }

    // 1. Grade
    const evaluation = await evaluateAnswer(question, answerText);

    // 2. Save
    const updatedSession = await stateManager.saveTurn(
      sessionId,
      question,
      answerText,
      evaluation
    );

    // 3. CHECK GAME OVER
    if (updatedSession.history.length >= MAX_QUESTIONS) {
      const totalScore = updatedSession.history.reduce(
        (sum, turn) => sum + turn.score,
        0
      );
      const averageScore = Math.round(
        totalScore / updatedSession.history.length
      );

      const finalReport = await generateFinalReport(
        updatedSession.history,
        updatedSession.jobDescription,
        updatedSession.gapAnalysis
      );

      const finalPayload = {
        ...finalReport,
        originalGapAnalysis: updatedSession.gapAnalysis,
      };

      await prisma.interview.create({
        data: {
          id: sessionId,
          jobDescription: updatedSession.jobDescription,
          finalScore: averageScore,
          finalFeedback: finalPayload,
          turns: {
            create: updatedSession.history.map((turn) => ({
              question: turn.question,
              answer: turn.answer,
              score: turn.score,
              feedback: turn.feedback,
              improvedAnswer: turn.betterAnswer,
              topic: turn.topic || "General",
              difficulty: turn.difficulty || "Medium",
            })),
          },
        },
      });

      await stateManager.deleteSession(sessionId);

      // ✅ SILENT FINISH
      // We return audio: null so the frontend just redirects without speaking.
      return res.json({
        evaluation,
        isFinished: true,
        finalReport: finalPayload,
        transcript: answerText,
        audio: null,
      });
    }

    // 4. NEXT QUESTION
    const queue = updatedSession.questionQueue;
    let nextQ;

    if (queue && queue.length > 0) {
      nextQ = queue[0];
    } else {
      nextQ = await getNextQuestion(updatedSession);
    }

    await stateManager.updateCurrentQuestion(sessionId, nextQ);

    // ✅ GENERATE AUDIO FOR NEXT QUESTION ONLY
    // We STRICTLY pass only `nextQ.question` to the generator.
    let audioBase64 = null;
    if (process.env.ENABLE_TTS === "true" && nextQ?.question) {
      try {
        console.time("TTS Generation");
        const audioBuffer = await generateAudio(nextQ.question);
        console.timeEnd("TTS Generation");

        if (audioBuffer) {
          audioBase64 = audioBuffer.toString("base64");
        }
      } catch (e) {
        console.error("TTS Error:", e);
      }
    } else {
      console.log("🔕 TTS Disabled (Saving Credits)");
    }

    return res.json({
      evaluation,
      isFinished: false,
      nextQuestion: nextQ,
      transcript: answerText,
      audio: audioBase64, // <--- Contains ONLY the question audio
    });
  } catch (error) {
    console.error("Submit Error:", error);
    res.status(500).json({ error: "Failed to process answer." });
  }
};
