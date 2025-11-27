import { createClient } from "redis";
import dotenv from "dotenv";
dotenv.config();

const client = createClient({
  url: process.env.REDIS_URL || "redis://localhost:6379",
});

client.on("error", (err) => console.error("Redis Client Error", err));
await client.connect();

export const stateManager = {
  // 1. Start Interview: Save Data
  async initSession(sessionId, data) {
    const sessionData = {
      jobDescription: data.jobDescription,
      questionQueue: data.initialQuestions,
      gapAnalysis: data.gapAnalysis,
      history: [],
      createdAt: new Date().toISOString(),
    };

    // Save as a simple JSON String (Compatible with all Redis versions)
    await client.set(`interview:${sessionId}`, JSON.stringify(sessionData));
    await client.expire(`interview:${sessionId}`, 86400); // 24h expiry
  },
  async deleteSession(sessionId) {
    try {
      console.log(`   🗑️ Deleting Redis Key: interview:${sessionId}`);
      await client.del(`interview:${sessionId}`);
      console.log("   ✅ Redis Key Deleted");
    } catch (e) {
      console.error("   ❌ Redis Delete Failed:", e.message);
    }
  },

  // 2. Get Full Session Data
  async getSession(sessionId) {
    const data = await client.get(`interview:${sessionId}`);
    return data ? JSON.parse(data) : null;
  },

  // 3. Save a Turn (Answer + Score) & Remove Question from Queue

  async saveTurn(sessionId, question, answer, evaluation) {
    // A. Fetch current state
    const sessionJson = await client.get(`interview:${sessionId}`);
    if (!sessionJson) return null;

    const session = JSON.parse(sessionJson);

    // B. Add to History
    session.history.push({
      question,
      answer,
      score: evaluation.score,
      betterAnswer: evaluation.betterAnswer, // <--- NEW
      feedback: evaluation.feedback,
      softSkillScore: null, // <--- Placeholder for later
      timestamp: new Date().toISOString(),
    });

    // C. Remove the First Question from Queue (if it exists)
    // Logic: If we just asked Q1, we remove Q1 so Q2 becomes the new "Next"
    if (session.questionQueue.length > 0) {
      session.questionQueue.shift(); // Remove the top item
    }

    // D. Save back to Redis
    await client.set(`interview:${sessionId}`, JSON.stringify(session));

    return session; // Return updated session
  },
};
