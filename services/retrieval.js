import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { prisma } from "../config/db.js";
import similarity from "compute-cosine-similarity";
import dotenv from "dotenv";
dotenv.config();

// 1. Initialize Models
const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001", // 👈 force v1
});

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);

// CHANGE 1: Use the stable model to prevent 404 errors
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-lite" });

// CHANGE 2: Removed the manual 'function cosineSimilarity' (Dead code)

/**
 * 🔍 Retrieve Context
 */
async function retrieveContext(question) {
  const qVector = await embeddings.embedQuery(question);
  const vectorString = `[${qVector.join(",")}]`;

  // Get Top 3 chunks
  const results = await prisma.$queryRawUnsafe(
    `SELECT content, metadata 
     FROM "Document" 
     ORDER BY embedding <=> $1::vector ASC 
     LIMIT 3`,
    vectorString,
  );

  return results;
}

/**
 * 🤖 The Judge: LLM Evaluation
 */
async function evaluateWithLLM(question, answer, contextString) {
  const prompt = `
    You are a strict Technical Interviewer.
    
    ### CONTEXT (Official Documentation):
    ${contextString.substring(0, 3000)}

    ### QUESTION:
    ${question}

    ### CANDIDATE ANSWER:
    "${answer}"

    ### GRADING RULES:
    1. If the answer is "gibberish", "random characters", or completely unrelated, SCORE = 0.
    2. If the answer is technically incorrect, SCORE = 0-20.
    3. If the answer is correct but vague, SCORE = 40-60.
    4. If the answer is correct and detailed, SCORE = 80-100.
    5. IGNORE spelling mistakes. Focus on Concepts.

    ### OUTPUT FORMAT (JSON ONLY):
    {
      "score": number,
      "feedback": "string (max 2 sentences)",
      "correctness": "string (Correct/Incorrect/Partial)"
      "betterAnswer": "string (A concise, ideal answer to this question)"
    }
  `;

  try {
    const result = await model.generateContent(prompt);
    const text = result.response.text();
    const jsonStr = text
      .replace(/```json/g, "")
      .replace(/```/g, "")
      .trim();
    return JSON.parse(jsonStr);
  } catch (e) {
    console.error("LLM Generation Failed:", e.message);
    return null;
  }
}

/**
 * 🚀 Main Function
 */
async function evaluateAnswer(question, userAnswer) {
  console.log("\n" + "=".repeat(60));
  console.log("🤖 HIRELY AI JUDGE");
  console.log("=".repeat(60));
  console.log(`❓ Q: ${question}`);
  console.log(`🗣️  A: ${userAnswer}`);

  try {
    // 1. Calculate Similarity
    console.log("\n📊 Calculating Semantic Similarity...");
    const [qVector, aVector] = await Promise.all([
      embeddings.embedQuery(question),
      embeddings.embedQuery(userAnswer),
    ]);

    const rawSimilarity = similarity(qVector, aVector) || 0;
    const similarityPercent = (rawSimilarity * 100).toFixed(1);

    // CHANGE 3: Uncommented this so you can debug!
    console.log(`   👉 Similarity Score: ${similarityPercent}%`);

    // 2. Gatekeeper
    if (rawSimilarity < 0.3) {
      console.log(
        `\n🛑 REJECTED: Similarity is below 30%. Answer is likely irrelevant.`,
      );
      const result = {
        score: 0,
        correctness: "Irrelevant",
        feedback: "Your answer appears to be off-topic.",
        betterAnswer:
          "N/A - The answer provided was unrelated to the technical question.",
      };
      printResult(result);
      return result;
    }

    console.log("   ✅ Passed Threshold (>30%).");

    // 3. Retrieve Context
    console.log("\n📚 Retrieving Official Documentation...");
    const contextDocs = await retrieveContext(question);

    if (!contextDocs || contextDocs.length === 0) {
      console.warn("   ⚠️ No context found in DB.");
    } else {
      console.log("   --------------------------------------------------");
      contextDocs.forEach((doc, i) => {
        const source = doc.metadata?.source || "Unknown";
        // Clean newlines for preview
        const preview = doc.content.replace(/\n/g, " ").substring(0, 80);
        console.log(`   [Chunk ${i + 1}] 🔗 ${source}`);
        console.log(`              📝 "${preview}..."`);
      });
      console.log("   --------------------------------------------------");
    }

    // Combine text for the LLM
    const contextString = contextDocs.map((d) => d.content).join("\n\n");

    // 4. LLM Grading
    console.log("\n⚖️  Sending to Gemini 1.5 for Grading...");
    const evaluation = await evaluateWithLLM(
      question,
      userAnswer,
      contextString || "No context provided.",
    );

    if (!evaluation) throw new Error("AI Service Unavailable");

    printResult(evaluation);
    return evaluation;
  } catch (error) {
    console.error("❌ Evaluation Error:", error.message);
    return { error: "System Error" };
  } finally {
    await prisma.$disconnect();
  }
}

function printResult(evaluation) {
  console.log("\n" + "-".repeat(60));
  console.log(`🏆 FINAL SCORE: ${evaluation.score}/100`);
  console.log(`📝 Result: ${evaluation.correctness}`);
  console.log(`💬 Feedback: ${evaluation.feedback}`);
  console.log(`✨ Better Answer: ${evaluation.betterAnswer}`);
  console.log("-".repeat(60));
}

// CLI Test Runner
if (process.argv[2]) {
  const q = process.argv[2];
  const a = process.argv[3];
  evaluateAnswer(q, a);
}

export { evaluateAnswer };
