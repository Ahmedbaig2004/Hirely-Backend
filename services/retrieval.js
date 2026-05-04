import { generateStructured, embedText, batchEmbedTexts } from "../config/gemini.js";
import { prisma } from "../config/db.js";
import similarity from "compute-cosine-similarity";
import { z } from "zod";
import dotenv from "dotenv";
dotenv.config();

/**
 * 🔍 Retrieve Context
 * Accepts an optional pre-computed qVector to avoid a duplicate embedding call.
 */
async function retrieveContext(question, existingQVector = null, categoryHint = null) {
  const qVector = existingQVector || (await embedText(question));
  const vectorString = `[${qVector.join(",")}]`;

  if (categoryHint) {
    const categoryResults = await prisma.$queryRawUnsafe(
      `SELECT content, metadata
       FROM "Document"
       WHERE metadata->>'category' = $2
       ORDER BY embedding <=> $1::vector ASC
       LIMIT 3`,
      vectorString,
      categoryHint,
    );

    return categoryResults;
  }

  const results = await prisma.$queryRawUnsafe(
    `SELECT content, metadata
     FROM "Document"
     ORDER BY embedding <=> $1::vector ASC
     LIMIT 3`,
    vectorString,
  );

  return results;
}

function toRecommendedDoc(doc) {
  if (!doc) return null;
  const source = doc.metadata?.source || doc.source || null;
  if (!source) return null;
  return {
    source,
    title: doc.metadata?.title || null,
    category: doc.metadata?.category || null,
    snippet: (doc.content || "").replace(/\s+/g, " ").trim().substring(0, 220),
  };
}

/**
 * 🤖 The Judge: LLM Evaluation
 */
const EvaluationSchema = z.object({
  isOnTopic: z.boolean(),
  relevanceScore: z.number().min(0).max(100),
  score: z.number().min(0).max(100),
  feedback: z.string(),
  correctness: z.enum(["Correct", "Incorrect", "Partial", "Irrelevant"]),
  betterAnswer: z.string(),
});

const NON_ANSWER_PHRASES = new Set([
  "hello",
  "hi",
  "hey",
  "no",
  "nope",
  "nah",
  "yes",
  "yeah",
  "yep",
  "ok",
  "okay",
  "hmm",
  "um",
  "uh",
  "idk",
  "i dont know",
  "i don't know",
  "not sure",
  "nothing",
  "none",
  "thank you",
  "thanks",
]);

const ANSWER_STOPWORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "about",
  "can",
  "do",
  "for",
  "give",
  "i",
  "is",
  "it",
  "me",
  "my",
  "of",
  "on",
  "or",
  "the",
  "to",
  "what",
  "with",
  "you",
  "your",
]);

const GENERIC_QUESTION_WORDS = new Set([
  "also",
  "common",
  "concept",
  "couple",
  "define",
  "describe",
  "difference",
  "example",
  "explain",
  "experience",
  "give",
  "interview",
  "mean",
  "moving",
  "overview",
  "tell",
  "use",
  "uses",
  "what",
  "when",
  "where",
  "why",
]);

const TOPIC_KEYWORD_GROUPS = [
  {
    triggers: ["aws", "amazon", "cloud", "s3"],
    terms: [
      "api",
      "archive",
      "autoscaling",
      "aws",
      "backup",
      "bucket",
      "cdn",
      "cloud",
      "cloudfront",
      "cloudwatch",
      "compute",
      "database",
      "dynamodb",
      "ebs",
      "ec2",
      "ecs",
      "eks",
      "file",
      "files",
      "iam",
      "instance",
      "lambda",
      "object",
      "objects",
      "rds",
      "region",
      "s3",
      "scalable",
      "server",
      "serverless",
      "storage",
      "static",
      "vpc",
    ],
  },
  {
    triggers: ["react", "component", "hook", "hooks", "jsx"],
    terms: [
      "component",
      "context",
      "effect",
      "hook",
      "hooks",
      "jsx",
      "memo",
      "props",
      "react",
      "render",
      "state",
      "virtual",
    ],
  },
  {
    triggers: ["sql", "database", "postgres", "postgresql"],
    terms: [
      "database",
      "foreign",
      "index",
      "join",
      "key",
      "postgres",
      "postgresql",
      "query",
      "record",
      "row",
      "sql",
      "table",
      "transaction",
    ],
  },
  {
    triggers: ["docker", "container", "devops", "kubernetes"],
    terms: [
      "build",
      "container",
      "deployment",
      "devops",
      "docker",
      "image",
      "kubernetes",
      "pipeline",
      "pod",
      "registry",
      "volume",
    ],
  },
];

const CATEGORY_ALIASES = [
  { category: "aws", terms: ["aws", "amazon", "s3", "lambda", "ec2", "iam"] },
  { category: "react", terms: ["react", "jsx", "hooks", "hook"] },
  { category: "nextjs", terms: ["nextjs", "next.js"] },
  { category: "nodejs", terms: ["node", "nodejs", "node.js", "express"] },
  { category: "typescript", terms: ["typescript", "ts"] },
  { category: "javascript", terms: ["javascript", "js"] },
  { category: "sql", terms: ["sql", "postgres", "postgresql"] },
  { category: "docker", terms: ["docker", "container"] },
  { category: "kubernetes", terms: ["kubernetes", "k8s"] },
  { category: "terraform", terms: ["terraform"] },
  { category: "python", terms: ["python"] },
  { category: "java", terms: ["java", "spring"] },
  { category: "system_design", terms: ["system design", "scalability"] },
];

function inferRetrievalCategory(question, roleContext = "") {
  const text = `${question} ${roleContext}`.toLowerCase();
  const tokens = new Set(meaningfulTokens(text));

  for (const { category, terms } of CATEGORY_ALIASES) {
    if (
      terms.some((term) =>
        term.includes(" ") ? text.includes(term) : tokens.has(term),
      )
    ) {
      return category;
    }
  }

  return null;
}

function normalizeAnswerText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^\w\s']/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function meaningfulTokens(text) {
  return normalizeAnswerText(text)
    .split(" ")
    .filter((token) => token && !ANSWER_STOPWORDS.has(token));
}

function importantQuestionTokens(text) {
  return meaningfulTokens(text).filter(
    (token) =>
      token.length > 2 &&
      !GENERIC_QUESTION_WORDS.has(token) &&
      !ANSWER_STOPWORDS.has(token),
  );
}

function expectedTopicTerms(question, roleContext = "") {
  const baseText = `${question} ${roleContext}`;
  const seedTokens = new Set(meaningfulTokens(baseText));
  const terms = new Set(importantQuestionTokens(question));

  for (const group of TOPIC_KEYWORD_GROUPS) {
    if (group.triggers.some((trigger) => seedTokens.has(trigger))) {
      group.terms.forEach((term) => terms.add(term));
    }
  }

  return terms;
}

function getTopicMismatchReason(answer, question, roleContext = "") {
  const answerTokens = meaningfulTokens(answer).filter(
    (token) => token.length > 2,
  );
  if (answerTokens.length < 3) return null;

  const expectedTerms = expectedTopicTerms(question, roleContext);
  if (expectedTerms.size === 0) return null;

  const hasTopicSignal = answerTokens.some((token) => expectedTerms.has(token));
  if (!hasTopicSignal) return "no question/topic keywords found";

  return null;
}

function getNonAnswerReason(answer, question) {
  const normalized = normalizeAnswerText(answer);
  if (!normalized) return "empty answer";
  if (NON_ANSWER_PHRASES.has(normalized)) return "generic non-answer phrase";

  const answerTokens = meaningfulTokens(answer);
  if (answerTokens.length < 2) return "too little answer content";

  const questionTokenSet = new Set(meaningfulTokens(question));
  const hasQuestionSignal = answerTokens.some(
    (token) => token.length > 2 && questionTokenSet.has(token),
  );

  if (normalized.split(" ").length < 4 && !hasQuestionSignal) {
    return "too short and unrelated";
  }

  return null;
}

function buildRejectedAnswerResult(reason) {
  return {
    score: 0,
    correctness: "Irrelevant",
    feedback:
      reason === "generic non-answer phrase" || reason === "empty answer"
        ? "Your answer did not contain enough content to evaluate."
        : "Your answer appears to be off-topic or too short to evaluate.",
    betterAnswer:
      "N/A - The answer provided did not meaningfully answer the technical question.",
  };
}

async function evaluateWithLLM(
  question,
  answer,
  contextString,
  roleContext,
  difficulty = "Medium",
) {
  const prompt = `
    You are a Technical Interviewer evaluating a candidate.

    ### ROLE CONTEXT:
    ${(roleContext || "").substring(0, 800)}

    ### IMPORTANT — CALIBRATE TO THE ROLE:
    Read the job description above carefully. Identify the seniority level (Junior, Mid, Senior, Lead, etc.) and the role requirements.
    - For JUNIOR roles: accept simplified, high-level explanations. A correct conceptual answer with basic terminology is strong (70-90). Don't expect deep internals, edge cases, or production-scale examples.
    - For MID roles: expect solid understanding with some practical depth. Correct answers with reasonable detail score 70-85.
    - For SENIOR roles: expect precise, detailed answers with real-world nuance, trade-offs, and edge cases for high scores (80-100).
    - Always grade relative to what is REASONABLE for the role level, not against an absolute expert standard.

    ### QUESTION DIFFICULTY: ${difficulty}
    Adjust your expectations based on difficulty vs. the role level above:
    - If this is a HARD question for a JUNIOR role (stretch territory): a partial answer showing awareness is good (50-70). A correct but incomplete answer is strong (70-85). Don't expect full expert knowledge.
    - If this is an EASY question for a SENIOR role: expect crisp, confident answers — vagueness on basics should score lower.
    - If the difficulty matches the role level: use the standard calibration above.

    ### CONTEXT (Official Documentation):
    ${contextString.substring(0, 3000)}

    ### QUESTION:
    ${question}

    ### CANDIDATE ANSWER:
    "${answer}"

    ### ON-TOPIC DECISION:
    Before scoring correctness, classify whether the answer actually addresses the question.
    If the answer is a greeting, refusal, personal story, unrelated topic, jargon from another technology, random text, or does not attempt the asked concept, return isOnTopic=false, relevanceScore=0, score=0, and correctness="Irrelevant".
    If the answer is on-topic but wrong, return isOnTopic=true and grade technical correctness normally.

    ### GRADING RULES:
    1. If the answer is "gibberish", "random characters", or completely unrelated, SCORE = 0.
    2. If the answer is technically incorrect, SCORE = 0-20.
    3. If the answer is partially correct or correct but vague for the role level, SCORE = 40-60.
    4. If the answer meets the expected depth for the role level, SCORE = 70-85.
    5. If the answer exceeds expectations for the role level, SCORE = 85-100.
    6. IGNORE spelling mistakes. Focus on Concepts.
    7. The candidate may answer in Roman Urdu (Urdu written in Latin script) mixed with English. This is normal code-switching — evaluate TECHNICAL CONTENT regardless of language. Roman Urdu words are NOT gibberish.

    ### OUTPUT FORMAT (JSON ONLY):
    {
      "isOnTopic": boolean,
      "relevanceScore": number,
      "score": number,
      "feedback": "string (max 2 sentences)",
      "correctness": "string (Correct/Incorrect/Partial/Irrelevant)"
      "betterAnswer": "string (A concise, ideal answer appropriate for this role level)"
    }
  `;

  try {
    const evaluation = await generateStructured(prompt, EvaluationSchema, {
      model: "gemini-2.5-flash",
    });
    if (!evaluation.isOnTopic || evaluation.relevanceScore < 25) {
      return {
        ...evaluation,
        isOnTopic: false,
        relevanceScore: Math.min(evaluation.relevanceScore ?? 0, 24),
        score: 0,
        correctness: "Irrelevant",
      };
    }
    return evaluation;
  } catch (e) {
    console.error("LLM Generation Failed:", e.message);
    return null;
  }
}

/**
 * 🚀 Main Function
 */
async function evaluateAnswer(
  question,
  userAnswer,
  roleContext,
  difficulty = "Medium",
) {
  console.log("\n" + "=".repeat(60));
  console.log("🤖 HIRELY AI JUDGE");
  console.log("=".repeat(60));
  console.log(`❓ Q: ${question}`);
  console.log(`🗣️  A: ${userAnswer}`);

  try {
    const nonAnswerReason = getNonAnswerReason(userAnswer, question);
    if (nonAnswerReason) {
      console.log(`\nREJECTED: ${nonAnswerReason}.`);
      const result = buildRejectedAnswerResult(nonAnswerReason);
      printResult(result);
      return result;
    }

    // 1. Embed question + answer sequentially to reduce rate-limit bursts
    console.log("\n📊 Calculating Semantic Similarity...");
    const qVector = await embedText(question);
    const aVector = await embedText(userAnswer);

    const rawSimilarity = similarity(qVector, aVector) || 0;
    const similarityPercent = (rawSimilarity * 100).toFixed(1);

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

    // 3. Retrieve Context — reuse qVector already computed above (no duplicate embed call)
    console.log("\n📚 Retrieving Official Documentation...");
    const categoryHint = inferRetrievalCategory(question, roleContext);
    if (categoryHint) console.log(`   Category filter: ${categoryHint}`);
    const contextDocs = await retrieveContext(question, qVector, categoryHint);

    if (!contextDocs || contextDocs.length === 0) {
      console.warn("   ⚠️ No context found in DB.");
    } else {
      console.log("   --------------------------------------------------");
      contextDocs.forEach((doc, i) => {
        const source = doc.metadata?.source || "Unknown";
        const preview = doc.content.replace(/\n/g, " ").substring(0, 80);
        console.log(`   [Chunk ${i + 1}] 🔗 ${source}`);
        console.log(`              📝 "${preview}..."`);
      });
      console.log("   --------------------------------------------------");
    }

    const contextString = contextDocs.map((d) => d.content).join("\n\n");
    const recommendedDoc = toRecommendedDoc(contextDocs[0]);

    // 4. LLM Grading
    console.log("\n⚖️  Sending to Gemini 3.0 for Grading...");
    const evaluation = await evaluateWithLLM(
      question,
      userAnswer,
      contextString || "No context provided.",
      roleContext,
      difficulty,
    );

    if (!evaluation) throw new Error("AI Service Unavailable");

    const result = { ...evaluation, recommendedDoc };
    printResult(result);
    return result;
  } catch (error) {
    const message = error?.message ? String(error.message) : "Unknown error";
    const isCritical =
      message.includes("429") ||
      message.includes("RESOURCE_EXHAUSTED") ||
      message.includes("500") ||
      message.includes("INTERNAL");
    console.error("❌ Evaluation Error:", message);
    if (isCritical) {
      throw new Error(message);
    }
    return { error: "System Error" };
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
  const ctx = process.argv[4] || "";
  evaluateAnswer(q, a, ctx);
}

async function evaluateAnswerBatch(turns, roleContext) {
  const nonAnswerReasons = turns.map((turn) =>
    getNonAnswerReason(turn.evaluationText || turn.answer, turn.question),
  );
  const allTexts = [];
  const validTurnIndexes = [];
  const vectorsByTurn = Array(turns.length).fill(null);

  for (let i = 0; i < turns.length; i++) {
    if (nonAnswerReasons[i]) continue;
    const turn = turns[i];
    validTurnIndexes.push(i);
    allTexts.push(turn.question);
    allTexts.push(turn.evaluationText || turn.answer);
  }

  if (allTexts.length > 0) {
    console.log(
      `\n📦 Batch-embedding ${allTexts.length} texts (${validTurnIndexes.length} evaluable turns) in 1 API call...`,
    );
  } else {
    console.log("\n📦 No evaluable answer content found. Skipping embeddings.");
  }
  const allVectors =
    allTexts.length > 0 ? await batchEmbedTexts(allTexts) : [];
  for (let i = 0; i < validTurnIndexes.length; i++) {
    vectorsByTurn[validTurnIndexes[i]] = {
      qVector: allVectors[i * 2],
      aVector: allVectors[i * 2 + 1],
    };
  }
  if (allTexts.length > 0) {
    console.log(`   ✅ Batch embedding complete.`);
  }

  const results = [];

  for (let i = 0; i < turns.length; i++) {
    const turn = turns[i];
    const qVector = vectorsByTurn[i]?.qVector;
    const aVector = vectorsByTurn[i]?.aVector;
    const question = turn.question;
    const userAnswer = turn.evaluationText || turn.answer;
    const difficulty = turn.difficulty || "Medium";

    console.log("\n" + "=".repeat(60));
    console.log("🤖 HIRELY AI JUDGE");
    console.log("=".repeat(60));
    console.log(`❓ Q: ${question}`);
    console.log(`🗣️  A: ${userAnswer}`);

    try {
      const nonAnswerReason = nonAnswerReasons[i];
      if (nonAnswerReason) {
        console.log(`\nREJECTED: ${nonAnswerReason}.`);
        const result = buildRejectedAnswerResult(nonAnswerReason);
        printResult(result);
        results.push(result);
        continue;
      }

      const rawSimilarity = similarity(qVector, aVector) || 0;
      const similarityPercent = (rawSimilarity * 100).toFixed(1);
      console.log(`\n📊 Similarity Score: ${similarityPercent}%`);

      if (rawSimilarity < 0.3) {
        console.log(`🛑 REJECTED: Below 30% threshold.`);
        const result = {
          score: 0,
          correctness: "Irrelevant",
          feedback: "Your answer appears to be off-topic.",
          betterAnswer:
            "N/A - The answer provided was unrelated to the technical question.",
        };
        printResult(result);
        results.push(result);
        continue;
      }

      console.log("   ✅ Passed Threshold (>30%).");

      console.log("\n📚 Retrieving Official Documentation...");
      const categoryHint = inferRetrievalCategory(question, roleContext);
      if (categoryHint) console.log(`   Category filter: ${categoryHint}`);
      const contextDocs = await retrieveContext(question, qVector, categoryHint);

      if (!contextDocs || contextDocs.length === 0) {
        console.warn("   ⚠️ No context found in DB.");
      } else {
        console.log("   --------------------------------------------------");
        contextDocs.forEach((doc, j) => {
          const source = doc.metadata?.source || "Unknown";
          const preview = doc.content.replace(/\n/g, " ").substring(0, 80);
          console.log(`   [Chunk ${j + 1}] 🔗 ${source}`);
          console.log(`              📝 "${preview}..."`);
        });
        console.log("   --------------------------------------------------");
      }

      const contextString = contextDocs.map((d) => d.content).join("\n\n");
      const recommendedDoc = toRecommendedDoc(contextDocs[0]);

      console.log("\n⚖️  Sending to Gemini for Grading...");
      const evaluation = await evaluateWithLLM(
        question,
        userAnswer,
        contextString || "No context provided.",
        roleContext,
        difficulty,
      );

      if (!evaluation) throw new Error("AI Service Unavailable");

      const result = { ...evaluation, recommendedDoc };
      printResult(result);
      results.push(result);
    } catch (error) {
      const message = error?.message ? String(error.message) : "Unknown error";
      const isCritical =
        message.includes("429") ||
        message.includes("RESOURCE_EXHAUSTED") ||
        message.includes("500") ||
        message.includes("INTERNAL");
      console.error("❌ Evaluation Error:", message);
      if (isCritical) throw new Error(message);
      results.push({ error: "System Error" });
    }
  }

  return results;
}

export { evaluateAnswer, evaluateAnswerBatch };
