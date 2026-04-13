import { Redis } from "@upstash/redis";
import dotenv from "dotenv";
dotenv.config();

const client = new Redis({
  url: process.env.REDIS_URL,
  token: process.env.REDIS_TOKEN,
});

export const stateManager = {
  // 1. Start Interview
  async initSession(sessionId, data) {
    const sessionData = {
      interviewType: data.interviewType || "JOB_SPECIFIC",
      config: data.config || null,
      jobDescription: data.jobDescription || null,
      questionQueue: data.initialQuestions,
      gapAnalysis: data.gapAnalysis || null,
      userId: data.userId,
      interviewerVoice: data.interviewerVoice || "female",
      // ✅ TRACK CURRENT QUESTION (Initialize with the first one)
      currentQuestion: data.initialQuestions[0],
      history: [],
      createdAt: new Date().toISOString(),
    };

    await client.set(`interview:${sessionId}`, sessionData, { ex: 86400 });
  },

  async deleteSession(sessionId) {
    await client.del(`interview:${sessionId}`);
  },

  async getSession(sessionId) {
    const data = await client.get(`interview:${sessionId}`);
    return data || null;
  },

  // 2. Update Current Question (Call this when generating Next Question)
  // Pass existingSession to skip the Redis GET when you already have the session in memory.
  async updateCurrentQuestion(sessionId, nextQuestionObj, existingSession = null) {
    const session = existingSession || await client.get(`interview:${sessionId}`);
    if (!session) return;

    session.currentQuestion = nextQuestionObj;

    await client.set(`interview:${sessionId}`, session);
  },

  // 3. Save Turn
  // evaluation and deliveryAnalysis are null during live interview (deferred to finalization).
  // evaluationText is the translated version (for Urdu answers) used for scoring.
  // detectedLanguage is stored so finalization can use the right filler word list.
  async saveTurn(sessionId, question, answer, evaluation, answerMode = "audio", deliveryAnalysis = null, evaluationText = null, detectedLanguage = "en") {
    const session = await client.get(`interview:${sessionId}`);
    if (!session) return null;

    const topic = session.currentQuestion?.topic || "General";
    const difficulty = session.currentQuestion?.difficulty || "Medium";

    session.history.push({
      question,
      answer,
      // Deferred scoring: score/feedback/betterAnswer are null during interview,
      // populated at finalization when all turns are evaluated in parallel.
      score: evaluation?.score ?? null,
      betterAnswer: evaluation?.betterAnswer ?? null,
      feedback: evaluation?.feedback ?? null,
      topic,
      difficulty,
      answerMode,
      deliveryAnalysis,
      // Store for deferred evaluation at finalization
      evaluationText: evaluationText ?? answer,
      detectedLanguage,
      timestamp: new Date().toISOString(),
    });

    if (session.questionQueue.length > 0) {
      session.questionQueue.shift();
    }

    await client.set(`interview:${sessionId}`, session);
    return session;
  },
};
