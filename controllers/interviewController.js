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

const prisma = new PrismaClient();
const MAX_QUESTIONS = 10;

/**
 * Initialize interview session
 * POST /api/init-interview
 */
export const initInterview = async (req, res) => {
  try {
    if (!req.file) throw new Error("No resume uploaded");

    // DIRECT BUFFER ACCESS
    const analysis = await generateInterviewContext(
      req.file.buffer,
      req.body.jobDescription
    );

    const sessionId = uuidv4();
    await stateManager.initSession(sessionId, {
      jobDescription: req.body.jobDescription,
      initialQuestions: analysis.questions,
      gapAnalysis: analysis.gapAnalysis,
    });

    res.json({
      sessionId,
      analysis,
      firstQuestion: analysis.questions[0],
    });
  } catch (e) {
    console.error("Init Error:", e);
    res.status(500).json({ error: "Init failed" });
  }
};

/**
 * Submit answer to interview question
 * POST /api/submit-answer
 */
export const submitAnswer = async (req, res) => {
  try {
    const { sessionId, question } = req.body;
    let answerText = "";

    // 🎤 Handle Audio or Text
    if (req.file) {
      console.log(`🎧 Processing Audio Answer (${req.file.size} bytes)...`);
      answerText = await transcribeAudio(req.file.buffer);
    } else if (req.body.answer) {
      answerText = req.body.answer;
    } else {
      return res.status(400).json({ error: "No audio or text provided." });
    }

    console.log(`🗣️  Final Answer: "${answerText}"`);

    // 📝 Grade
    const evaluation = await evaluateAnswer(question, answerText);

    const updatedSession = await stateManager.saveTurn(
      sessionId,
      question,
      answerText,
      evaluation
    );

    // 🏁 Game Over Check
    if (updatedSession.history.length >= MAX_QUESTIONS) {
      console.log("🏁 Interview Finished. Generating Report...");
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
        updatedSession.gapAnalysis // <--- PASS IT HERE
      );

      // Save to Postgres
      await prisma.interview.create({
        data: {
          id: sessionId,
          jobDescription: updatedSession.jobDescription,
          finalScore: averageScore,
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

      // Cleanup Redis
      await stateManager.deleteSession(sessionId);

      return res.json({
        evaluation,
        isFinished: true,
        finalReport,
        transcript: answerText,
      });
    }

    // ➡️ Next Question
    const queue = updatedSession.questionQueue;
    let nextQ;

    if (queue && queue.length > 0) {
      nextQ = queue[0];
    } else {
      nextQ = await getNextQuestion(updatedSession);
    }

    return res.json({
      evaluation,
      isFinished: false,
      nextQuestion: nextQ,
      transcript: answerText,
    });
  } catch (error) {
    console.error("❌ Answer Submission Error:", error);
    res.status(500).json({ error: "Failed to process answer." });
  }
};
