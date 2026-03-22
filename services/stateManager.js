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
      jobDescription: data.jobDescription,
      questionQueue: data.initialQuestions,
      gapAnalysis: data.gapAnalysis,
      userId: data.userId, 
      // ✅ TRACK CURRENT QUESTION (Initialize with the first one)
      currentQuestion: data.initialQuestions[0],
      history: [],
      createdAt: new Date().toISOString(),
    };

    await client.set(`interview:${sessionId}`, sessionData);
    await client.expire(`interview:${sessionId}`, 86400);
  },

  async deleteSession(sessionId) {
    await client.del(`interview:${sessionId}`);
  },

  async getSession(sessionId) {
    const data = await client.get(`interview:${sessionId}`);
    return data || null;
  },

  // 2. Update Current Question (Call this when generating Next Question)
  async updateCurrentQuestion(sessionId, nextQuestionObj) {
    const session = await client.get(`interview:${sessionId}`);
    if (!session) return;

    session.currentQuestion = nextQuestionObj;

    await client.set(`interview:${sessionId}`, session);
  },

  // 3. Save Turn
  async saveTurn(sessionId, question, answer, evaluation, answerMode = "audio", deliveryAnalysis = null) {
    const session = await client.get(`interview:${sessionId}`);
    if (!session) return null;

    const topic = session.currentQuestion?.topic || "General";
    const difficulty = session.currentQuestion?.difficulty || "Medium";

    session.history.push({
      question,
      answer,
      score: evaluation.score,
      betterAnswer: evaluation.betterAnswer,
      feedback: evaluation.feedback,
      topic,
      difficulty,
      answerMode,
      deliveryAnalysis,
      timestamp: new Date().toISOString(),
    });

    if (session.questionQueue.length > 0) {
      session.questionQueue.shift();
    }

    await client.set(`interview:${sessionId}`, session);
    return session;
  },
};
