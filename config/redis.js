import { createClient } from "redis";

async function testRedis() {
  const redis = createClient({
    url: process.env.REDIS_URL, // or use container name if running inside Docker
  });

  redis.on("error", (err) => console.error("Redis Error:", err));

  await redis.connect();

  const interviewId = "1234";

  // Save Q&A
  await redis.hSet(`interview:${interviewId}`, {
    q1: JSON.stringify({
      question: "What is JS?",
      answer: "A language",
      evaluation: "Good",
    }),
    q2: JSON.stringify({
      question: "What is Redis?",
      answer: "In-memory DB",
      evaluation: "Excellent",
    }),
  });

  // Fetch back
  const data = await redis.hGetAll(`interview:${interviewId}`);
  console.log("Q&A from Redis:", data);

  // Clean up
  await redis.del(`interview:${interviewId}`);
  await redis.quit();
}

testRedis();
