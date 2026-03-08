import { createClient } from "redis";
import dotenv from "dotenv";
dotenv.config();

const client = createClient({
  url: process.env.REDIS_URL || "redis://localhost:6379",
});

client.on("error", (err) => console.error("Redis Client Error", err));
await client.connect();

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

    await client.set(`interview:${sessionId}`, JSON.stringify(sessionData));
    await client.expire(`interview:${sessionId}`, 86400);
  },

  async deleteSession(sessionId) {
    await client.del(`interview:${sessionId}`);
  },

  async getSession(sessionId) {
    const data = await client.get(`interview:${sessionId}`);
    return data ? JSON.parse(data) : null;
  },

  // 2. Update Current Question (Call this when generating Next Question)
  async updateCurrentQuestion(sessionId, nextQuestionObj) {
    const sessionJson = await client.get(`interview:${sessionId}`);
    if (!sessionJson) return;
    const session = JSON.parse(sessionJson);

    session.currentQuestion = nextQuestionObj; // Store full object {question, topic, difficulty}

    await client.set(`interview:${sessionId}`, JSON.stringify(session));
  },

  // 3. Save Turn
  async saveTurn(sessionId, question, answer, evaluation) {
    const sessionJson = await client.get(`interview:${sessionId}`);
    if (!sessionJson) return null;
    const session = JSON.parse(sessionJson);

    // ✅ GET METADATA from the stored currentQuestion
    // Fallback to "Unknown" if something weird happens
    const topic = session.currentQuestion?.topic || "General";
    const difficulty = session.currentQuestion?.difficulty || "Medium";

    session.history.push({
      question,
      answer,
      score: evaluation.score,
      betterAnswer: evaluation.betterAnswer,
      feedback: evaluation.feedback,
      topic, // <--- Save Topic
      difficulty, // <--- Save Difficulty
      timestamp: new Date().toISOString(),
    });

    // Remove from queue if it exists
    if (session.questionQueue.length > 0) {
      session.questionQueue.shift();
    }

    await client.set(`interview:${sessionId}`, JSON.stringify(session));
    return session;
  },
};
