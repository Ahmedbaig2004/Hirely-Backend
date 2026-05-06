import { generateStructured } from "../config/gemini.js";
import { z } from "zod";
import dotenv from "dotenv";
import { getVoiceAnalysesForInterview } from "./voiceAnalysisHelper.js";

dotenv.config();

const W_AUDIO = 0.5;
const W_VIDEO = 0.5;

const NextQuestionSchema = z.object({
  bridge: z
    .string()
    .describe(
      "A short truthful transition that reacts to the candidate's last answer, or an empty string when no truthful transition is needed",
    ),
  question: z.string(),
  topic: z.string(),
  difficulty: z.string(),
  reason: z.string(),
});

const PRIMARY_NEXT_QUESTION_TIMEOUT_MS = 15000;

function withTimeout(promise, timeoutMs, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error(`${label} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

function normalizeWhitespace(text = "") {
  return String(text).replace(/\s+/g, " ").trim();
}

function dedupeAdjacentClauses(text = "") {
  const normalized = normalizeWhitespace(text);
  if (!normalized) return "";

  const parts = normalized
    .split(/(?<=[.!?])\s+/)
    .map((part) => normalizeWhitespace(part))
    .filter(Boolean);

  const deduped = [];
  for (const part of parts) {
    if (deduped.length === 0) {
      deduped.push(part);
      continue;
    }
    const prev = deduped[deduped.length - 1].toLowerCase();
    if (prev === part.toLowerCase()) continue;
    deduped.push(part);
  }

  return deduped.join(" ");
}

function cleanTranscriptForAdaptiveUse(text = "") {
  let cleaned = dedupeAdjacentClauses(text);

  cleaned = cleaned
    .replace(/\b(uh|um|hmm|mmm|you know|like)\b/gi, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!cleaned) return normalizeWhitespace(text);
  return cleaned;
}

function classifyAnswerTone(answer = "") {
  const normalized = normalizeWhitespace(answer).toLowerCase();
  if (!normalized) return "empty";

  const negativePatterns = [
    /^(no|nope|nah)\b/,
    /^(i do not|i don't|did not|didn't)\b/,
    /\bi (do not|don't|did not|didn't|can't|cannot|couldn't)\b/,
    /\b(i do not know|i don't know|not sure|unsure|no idea)\b/,
    /\bnever used\b/,
    /\bhaven't worked with\b/,
    /\bnot really\b/,
  ];

  if (negativePatterns.some((pattern) => pattern.test(normalized))) {
    return "negative";
  }

  return "neutral_or_positive";
}

function isOverlyPositiveBridge(bridge = "") {
  const normalized = normalizeWhitespace(bridge).toLowerCase();
  if (!normalized) return false;

  return [
    /\bgood\b/,
    /\bgreat\b/,
    /\bexcellent\b/,
    /\bstrong\b/,
    /\bwell done\b/,
    /\bnice\b/,
    /\bthat makes sense\b/,
  ].some((pattern) => pattern.test(normalized));
}

function isGenericBridge(bridge = "") {
  const normalized = normalizeWhitespace(bridge)
    .toLowerCase()
    .replace(/[.!?]+$/g, "");
  if (!normalized) return false;

  return [
    /^thanks?( for (that|sharing|sharing that|your answer|the answer))?$/,
    /^thank you( for (that|sharing|sharing that|your answer|the answer))?$/,
    /^got it$/,
    /^okay$/,
    /^ok$/,
    /^i see$/,
    /^understood$/,
    /^interesting$/,
    /^that makes sense$/,
  ].some((pattern) => pattern.test(normalized));
}

function buildFallbackBridge(currentTurn, payload, lastTurn) {
  const answer = normalizeWhitespace(currentTurn?.answer || "");
  const currentTopic = normalizeWhitespace(payload?.topic || "");
  const previousTopic = normalizeWhitespace(lastTurn?.topic || "");
  const answerTone = classifyAnswerTone(answer);

  if (!answer) return "";
  if (answerTone === "negative") {
    if (currentTopic && previousTopic && currentTopic !== previousTopic) {
      return "Okay, let's try a different angle.";
    }
    return "Okay.";
  }
  if (currentTopic && previousTopic && currentTopic === previousTopic) {
    return "Got it, let's build on that.";
  }
  if (currentTopic && previousTopic && currentTopic !== previousTopic) {
    return "Okay, let's shift slightly.";
  }
  return "Got it.";
}

function stripBridgePrefix(question = "", bridge = "") {
  const normalizedQuestion = normalizeWhitespace(question);
  const normalizedBridge = normalizeWhitespace(bridge);
  if (!normalizedQuestion || !normalizedBridge) return normalizedQuestion;

  const lowerQuestion = normalizedQuestion.toLowerCase();
  const lowerBridge = normalizedBridge.toLowerCase();

  if (lowerQuestion === lowerBridge) return "";
  if (lowerQuestion.startsWith(`${lowerBridge} `)) {
    return normalizedQuestion.slice(normalizedBridge.length).trim();
  }

  const bridgeLead = lowerBridge.replace(/[.!?]+$/g, "");
  const questionLead = lowerQuestion.replace(/[.!?]+$/g, "");
  if (questionLead.startsWith(bridgeLead) && bridgeLead.length > 12) {
    return normalizedQuestion.slice(normalizedBridge.length).trim();
  }

  return normalizedQuestion;
}

function stripGenericQuestionPrefix(question = "") {
  return normalizeWhitespace(question)
    .replace(
      /^(thanks?(?: for (?:that|sharing|sharing that|your answer|the answer))?|thank you(?: for (?:that|sharing|sharing that|your answer|the answer))?|got it|okay|ok|i see|understood|interesting|that makes sense)[.!?,]?\s+/i,
      "",
    )
    .trim();
}

function sanitizeNextQuestionPayload(
  payload,
  currentTurn = null,
  lastTurn = null,
) {
  if (!payload || typeof payload !== "object") return payload;

  let bridge = dedupeAdjacentClauses(payload.bridge || "");
  let question = stripBridgePrefix(
    dedupeAdjacentClauses(payload.question || ""),
    bridge,
  );
  question = stripGenericQuestionPrefix(question);
  const answerTone = classifyAnswerTone(currentTurn?.answer || "");

  if (
    answerTone === "negative" &&
    (isOverlyPositiveBridge(bridge) || isGenericBridge(bridge))
  ) {
    bridge = buildFallbackBridge(currentTurn, payload, lastTurn);
  }

  if (answerTone !== "negative" && isGenericBridge(bridge)) {
    bridge = "";
  }

  if (!bridge && answerTone === "negative") {
    bridge = buildFallbackBridge(currentTurn, payload, lastTurn);
  }

  question = question
    .replace(
      /\b(can you explain what an? .+? is and (why|how) .+? used\??)/i,
      (match) => match,
    )
    .trim();

  if (
    /\bbasic arithmetic operators\b/i.test(question) ||
    /\bsimple expression\b/i.test(question) ||
    /\bwhat is an if statement\b/i.test(question) ||
    /\bwhat are .* operators\b/i.test(question)
  ) {
    question = question
      .replace(
        /Can you describe some of the basic arithmetic operators in Python, such as addition, subtraction, multiplication, and division\?/i,
        "Can you walk me through how Python arithmetic operators show up in real application logic or data processing work?",
      )
      .replace(
        /How would you use them in a simple expression\?/i,
        "Can you describe a simple real-world situation where you would use them and what you would be checking or calculating?",
      )
      .replace(
        /What is an if statement and how would you use it to make a decision in your code\?/i,
        "How do conditional branches help you make decisions in a Python program, and where have you used that kind of logic in practice?",
      )
      .replace(
        /Provide a very simple example\.?/i,
        "You can explain it in words rather than writing code.",
      )
      .trim();
  }

  return {
    ...payload,
    bridge,
    question,
  };
}

const FinalReportSchema = z.object({
  decision: z.enum(["Strong Hire", "Hire", "Weak Hire", "No Hire"]),
  technicalLevel: z
    .string()
    .describe("Estimated seniority (e.g. Junior, Mid, Senior)"),
  summary: z
    .string()
    .describe("Professional analysis of the candidate's performance"),
  strengths: z.array(z.string()),
  weaknesses: z.array(z.string()),
  recommendations: z
    .string()
    .describe("Advice for both technical and communication growth"),
});

const VoiceInsightsSchema = z.object({
  insights: z
    .array(z.string())
    .describe(
      "4-6 specific, actionable, non-repetitive observations about the candidate's communication style across the full interview",
    ),
});

/**
 * Use AI to synthesize raw voice metrics from all turns into coherent,
 * non-repetitive feedback. One LLM call covers the whole interview.
 */
async function generateAIVoiceInsights(voiceAnalyses) {
  if (!voiceAnalyses || voiceAnalyses.length === 0) return [];

  const turnSummaries = voiceAnalyses.map((v, i) => {
    const shapExplanations = v.rawFeatures?.shapExplanations || [];
    const hasShap = shapExplanations.length > 0;

    const base = {
      turn: i + 1,
      confidenceScore: Math.round((v.confidenceLevel || 0) * 100),
      confidenceLabel: v.confidenceLabelText || "Unknown",
      wpm: Math.round(v.wordsPerMinute || 0),
    };

    if (hasShap) {
      return {
        ...base,
        topFactors: shapExplanations.slice(0, 3).map((e) => ({
          feature: e.label,
          impact: e.direction,
          explanation: e.explanation,
        })),
      };
    }

    return {
      ...base,
      pauseRatioPct: Math.round((v.pauseRatio || 0) * 100),
      jitter: parseFloat((v.jitter || 0).toFixed(3)),
      shimmer: parseFloat((v.shimmer || 0).toFixed(3)),
      energy: parseFloat((v.energyLevel || 0).toFixed(3)),
    };
  });

  const hasShapData = turnSummaries.some((t) => t.topFactors);

  const prompt = `You are an expert communication coach reviewing ML-powered voice analysis from a job interview.

ANALYSIS PER QUESTION:
${JSON.stringify(turnSummaries, null, 2)}

${
  hasShapData
    ? `Each turn includes topFactors — the acoustic features the model identified as most impactful on the confidence score, whether each factor increased or decreased the score, and a coaching tip.`
    : `REFERENCE RANGES: WPM 100-160 ideal. Jitter <0.02 stable. Shimmer <0.08 stable. Pause ratio <20% fluent. Energy >0.05 good projection.`
}

TASK: Write 4-6 concise, actionable observations synthesizing the candidate's communication patterns across the WHOLE interview.

Rules:
- ${hasShapData ? "Use the SHAP-identified factors as your primary evidence — these are what the model actually measured" : "Use the metrics and reference ranges as your evidence"}
- If the same factor appears across multiple turns, treat it as a consistent pattern (stronger evidence)
- If a factor appears positive in some turns and negative in others, note the inconsistency
- Lead with strengths, then areas to improve
- Be specific and actionable (e.g. "try pausing between key points" not "work on pacing")
- Write in second person ("e.g .You maintained steady projection throughout")
- Keep each observation to one sentence
- Do not refer to specific turn numbers`;

  try {
    const result = await generateStructured(prompt, VoiceInsightsSchema);
    return result.insights;
  } catch (e) {
    console.warn("⚠️ AI voice insights generation failed:", e.message);
    return [];
  }
}

/**
 * Adaptive Questioning Logic
 *
 * @param {object} session - Current Redis session
 * @param {object} [currentTurn] - The turn being answered right now { question, answer }.
 *   When provided, question generation runs in parallel with evaluation — Gemini reads
 *   the raw transcript directly instead of waiting for a numeric score.
 */
export async function getNextQuestion(session, currentTurn = null) {
  const historyText = session.history
    .map(
      (h) =>
        `Q: ${h.question}\nA: ${h.answer}\nScore: ${
          h.score == null ? "Not scored yet" : `${h.score}/100`
        }\nDifficulty: ${h.difficulty || "Medium"}\nTopic: ${h.topic || "General"}`,
    )
    .join("\n---\n");

  const lastTurn = session.history[session.history.length - 1];
  const lastDifficulty = lastTurn?.difficulty || "Medium";
  const interviewType = session.interviewType || "JOB_SPECIFIC";
  const config = session.config || {};
  const selectedDifficulty = config.difficulty || "Medium";
  const interviewMode = session.interviewMode || "audio";

  const currentTurnSection = currentTurn
    ? `CURRENT QUESTION (just answered — transcript below, not yet scored):
Q: ${currentTurn.question}
A: ${cleanTranscriptForAdaptiveUse(currentTurn.answer)}
Difficulty so far: ${lastDifficulty}

Use this transcript to choose the next question. For fixed-difficulty modes, adapt only the content, not the difficulty.`
    : `LAST QUESTION was difficulty: ${lastDifficulty}. Score is ${
        lastTurn?.score == null ? "not available yet" : `${lastTurn.score}/100`
      }.`;

  let contextSection;
  let taskDescription;
  let followUpStyle;

  if (interviewType === "TECHNICAL") {
    const topicsCovered = session.history
      .map((h) => h.topic)
      .filter(Boolean)
      .join(", ");
    contextSection = `INTERVIEW TYPE: Technical — ${config.stack || "General"} (${config.difficulty || "Medium"} level)
Topics already covered: ${topicsCovered || "(none yet)"}`;
    taskDescription = `Generate the NEXT technical question about ${config.stack || "the stack"}.
- Prefer a real follow-up if the candidate said something worth probing deeper, clarifying, or challenging.
- Move to a new topic only when the previous area feels complete, weakly answered, or not worth drilling into further.
- Assume the candidate already knows the most basic syntax of ${config.stack || "the stack"}.
- Do NOT ask tutorial-style questions about primitive syntax, arithmetic operators, basic keywords, or classroom definitions.
- Do not ask the exact same question twice.`;
    followUpStyle = `TECHNICAL FOLLOW-UP STYLE:
- If the candidate named a tool, concept, architecture decision, tradeoff, or project detail, prefer drilling into that naturally.
- Good human follow-ups often ask "why", "how", "what happens if", or "can you walk me through".
- A new topic is fine, but only after a natural transition.
- This is a ${interviewMode} interview, so ask for explanation, reasoning, tradeoffs, debugging thought process, architecture, or a verbal walkthrough.
- Do NOT ask the candidate to write code, provide a snippet, implement a function, or give a runnable example.
- If you want an example, ask for a simple verbal scenario or pseudocode-level explanation in words.`;
  } else if (interviewType === "BEHAVIORAL") {
    const competenciesCovered = session.history
      .map((h) => h.topic)
      .filter(Boolean)
      .join(", ");
    contextSection = `INTERVIEW TYPE: Behavioral — STAR method (${config.difficulty || "Medium"} level)
Competencies already covered: ${competenciesCovered || "(none yet)"}`;
    taskDescription = `Generate the NEXT behavioral question expecting a STAR-method answer.
- Prefer a real follow-up if the candidate left out a key STAR element, gave a vague example, or mentioned something worth unpacking.
- Move to a new competency only when the previous story feels sufficiently explored.
- Do not ask the exact same question twice.`;
    followUpStyle = `BEHAVIORAL FOLLOW-UP STYLE:
- If the candidate gave an incomplete STAR answer, ask for the missing piece naturally.
- Good human follow-ups probe ownership, tradeoffs, conflict handling, results, and lessons learned.
- A new competency is fine, but only after a natural transition.`;
  } else {
    contextSection = `JOB CONTEXT: ${(session.jobDescription || "").substring(0, 500)}...`;
    taskDescription = `Generate the NEXT follow-up question.
- Prefer probing the candidate's actual last answer, especially if they mentioned a project, skill, gap, uncertainty, or partial answer worth exploring.
- Move to a new JD area only when that feels more natural than drilling deeper.
- Follow the difficulty rules above strictly.`;
    followUpStyle = `JOB-SPECIFIC FOLLOW-UP STYLE:
- If the candidate hinted at a relevant experience, ask them to expand on it naturally.
- If they exposed a gap, uncertainty, or weak explanation, probe it respectfully.
- If you switch topics, make the transition feel intentional, not abrupt.`;
  }

  if (interviewType === "TECHNICAL") {
    const topicsCovered = session.history
      .map((h) => h.topic)
      .filter(Boolean)
      .join(", ");
    contextSection = `INTERVIEW TYPE: Technical - ${config.stack || "General"} (${selectedDifficulty} level)
Topics already covered: ${topicsCovered || "(none yet)"}`;
    taskDescription = `Generate the NEXT technical question about ${config.stack || "the stack"}.
- Keep the difficulty exactly "${selectedDifficulty}". Do not increase or decrease it.
- Adapt by choosing the most useful follow-up or next topic based on the candidate's transcript.
- Prefer probing the previous answer before switching topics, unless the previous area is exhausted or too weak to continue productively.
- Assume the candidate already knows the most basic syntax of ${config.stack || "the stack"}.
- Do NOT ask tutorial-style questions about primitive syntax, arithmetic operators, basic keywords, or classroom definitions.
- Do not ask the exact same question twice.
- Set the "difficulty" field to exactly "${selectedDifficulty}".`;
  } else if (interviewType === "BEHAVIORAL") {
    const competenciesCovered = session.history
      .map((h) => h.topic)
      .filter(Boolean)
      .join(", ");
    contextSection = `INTERVIEW TYPE: Behavioral - STAR method (${selectedDifficulty} level)
Competencies already covered: ${competenciesCovered || "(none yet)"}`;
    taskDescription = `Generate the NEXT behavioral question expecting a STAR-method answer.
- Keep the difficulty exactly "${selectedDifficulty}". Do not increase or decrease it.
- Adapt by probing the most useful competency, weakness, missing STAR element, or follow-up area from the candidate's transcript.
- Prefer drilling deeper into the candidate's current story before jumping to a fresh competency.
- Do not ask the exact same question twice.
- Set the "difficulty" field to exactly "${selectedDifficulty}".`;
  }

  const modeDifficultyRules =
    interviewType === "TECHNICAL" || interviewType === "BEHAVIORAL"
      ? `FIXED DIFFICULTY OVERRIDE:
    - Ignore any instruction above that says to increase or decrease difficulty.
    - This mode uses the user's selected difficulty, so the next question difficulty must be exactly "${selectedDifficulty}".
    - Adapt only the content: topic, competency, follow-up angle, specificity, or depth within that same difficulty.`
      : `JOB-SPECIFIC DIFFICULTY:
    - Difficulty may increase, decrease, or stay the same based on the candidate's answer quality and the JD requirements.`;

  const prompt = `
    You are a warm, sharp, highly human interviewer.
    ${contextSection}

    INTERVIEW HISTORY (previous turns, scored):
    ${historyText || "(No previous turns yet)"}

    ${currentTurnSection}

    ADAPTIVE DIFFICULTY RULES:
    - Strong answer (clear, correct, detailed): you MAY increase difficulty by ONE level (Easy→Medium, Medium→Hard). Never jump two levels.
    - Weak answer (vague, incorrect, "I don't know"): DECREASE difficulty or stay the same. Do NOT keep asking Hard questions after a poor answer.
    - Always set the "difficulty" field to exactly one of: "Easy", "Medium", or "Hard".

    ${modeDifficultyRules}

    CONVERSATIONAL STYLE RULES:
    - Sound like a real interviewer, not a question generator.
    - Use a bridge only when it can truthfully react to something specific in the candidate's last answer.
    - Keep any bridge short: one brief sentence or phrase, not a long summary.
    - Put that bridge in the "bridge" field.
    - If there is no specific, truthful acknowledgment to make, set "bridge" to an empty string and ask the next question directly.
    - Do not flatter excessively, do not sound scripted, and do not mention internal scoring logic.
    - Make the bridge truthful to the answer quality: if the candidate says "no", "I don't know", or gives a weak/negative answer, do not praise it with words like "good", "great", or "strong".
    - Avoid generic filler bridges like "Thanks for that", "Thanks for sharing", "Got it", "Okay", "Interesting", or "That makes sense" unless they are directly followed by a specific reference to the answer.
    - For weak or negative answers, use a short neutral bridge such as "Okay." only when a bridge is needed.
    - Never say "final question".
    - Do not over-quote unusual transcript fragments unless they are clearly central and reliable.
    - If the transcript contains a noisy or oddly specific phrase, prefer a general acknowledgment over repeating it literally.
    - Respect the interview mode: the candidate is answering in ${interviewMode}, not in a code editor.
    - For technical interviews, do not ask for runnable code, pasted code snippets, or coding-exercise style implementations.
    - For technical interviews, avoid textbook phrasing like "what is an if statement", "list basic operators", or "write a simple expression".

    ${followUpStyle || ""}

    TASK: ${taskDescription}

    OUTPUT RULES:
    - Put a brief truthful acknowledgment / transition in the "bridge" field, or use an empty string if the candidate's answer gives you nothing specific to acknowledge.
    - Put only the actual next interview question in the "question" field.
    - The "bridge" should reference or react to the candidate's last answer when possible.
    - If the candidate's last answer gives you very little to reference, leave "bridge" empty instead of using generic thanks.
    - The "reason" field is internal and should explain why this is the best next move.
  `;

  try {
    const result = await withTimeout(
      generateStructured(prompt, NextQuestionSchema, {
        model: "gemini-2.5-flash-lite",
        temperature: 0.3,
        maxOutputTokens: 900,
        thinkingConfig: { thinkingBudget: 0 },
      }),
      PRIMARY_NEXT_QUESTION_TIMEOUT_MS,
      "Adaptive question generation with gemini-2.5-flash-lite",
    );
    return sanitizeNextQuestionPayload(result, currentTurn, lastTurn);
  } catch (error) {
    console.warn(
      "Adaptive question generation failed on gemini-2.5-flash-lite, retrying with gemini-2.5-flash:",
      error.message,
    );
    const fallback = await generateStructured(prompt, NextQuestionSchema, {
      model: "gemini-2.5-flash",
      temperature: 0.3,
      maxOutputTokens: 300,
      thinkingConfig: { thinkingBudget: 0 },
    });
    return sanitizeNextQuestionPayload(fallback, currentTurn, lastTurn);
  }
}

/**
 * Final Report Generation with Integrated Voice Insights
 * @param {string[]} prefetchedVoiceData - Optional raw voice data already fetched from Redis.
 *   When provided, skips a redundant Redis read inside this function.
 */
export async function generateFinalReport(
  sessionId,
  history,
  jobDescription,
  gapAnalysis,
  prefetchedVoiceData = null,
  interviewType = "JOB_SPECIFIC",
  config = null,
  prefetchedVideoData = null,
) {
  // 1. Calculate Technical Score (difficulty-adjusted)
  let roleSeniority = "mid";
  if (interviewType === "JOB_SPECIFIC" && jobDescription) {
    const jdLower = jobDescription.toLowerCase();
    roleSeniority =
      jdLower.includes("senior") ||
      jdLower.includes("lead") ||
      jdLower.includes("principal")
        ? "senior"
        : jdLower.includes("junior") ||
            jdLower.includes("entry") ||
            jdLower.includes("graduate") ||
            jdLower.includes("intern")
          ? "junior"
          : "mid";
  } else if (config?.difficulty) {
    roleSeniority =
      config.difficulty === "Hard"
        ? "senior"
        : config.difficulty === "Easy"
          ? "junior"
          : "mid";
  }

  const difficultyRank = { Easy: 0, Medium: 1, Hard: 2 };
  const roleBaseRank = { junior: 0, mid: 1, senior: 2 };

  function adjustScoreForDifficulty(rawScore, difficulty) {
    const gap = (difficultyRank[difficulty] ?? 1) - roleBaseRank[roleSeniority];
    if (gap > 0) {
      const floored = Math.max(rawScore, 50);
      return rawScore >= 70 ? Math.min(rawScore + 10, 100) : floored;
    }
    return rawScore;
  }

  const technicalScore =
    history.length > 0
      ? history.reduce(
          (sum, turn) =>
            sum +
            adjustScoreForDifficulty(turn.score, turn.difficulty || "Medium"),
          0,
        ) / history.length
      : 0;

  // 1b. Calculate Delivery Score (text modality)
  const turnsWithDelivery = history.filter((t) => t.deliveryAnalysis);
  const deliveryScore =
    turnsWithDelivery.length > 0
      ? turnsWithDelivery.reduce(
          (sum, t) => sum + t.deliveryAnalysis.deliveryScore,
          0,
        ) / turnsWithDelivery.length
      : null;

  // 2. Fetch interpreted Voice Data
  let voiceAnalyses = [];
  try {
    if (prefetchedVoiceData !== null) {
      console.log(`⏳ Using pre-fetched voice data for: ${sessionId}`);
      voiceAnalyses = prefetchedVoiceData;
    } else {
      console.log(`⏳ Extracting vocal insights for: ${sessionId}`);
      voiceAnalyses = await getVoiceAnalysesForInterview(
        sessionId,
        history.length,
      );
    }
  } catch (e) {
    console.warn("⚠️ Voice analysis retrieval failed:", e.message);
  }

  // 3. Aggregate Communication Patterns
  const hasVoiceData = voiceAnalyses.length > 0;
  const voiceScore = hasVoiceData
    ? voiceAnalyses.reduce((sum, v) => sum + v.confidenceLevel * 100, 0) /
      voiceAnalyses.length
    : null;

  console.log(
    `🤖 Generating AI voice insights for ${voiceAnalyses.length} turns...`,
  );
  const allVocalInsights = await generateAIVoiceInsights(voiceAnalyses);

  const deliveryInsights =
    turnsWithDelivery.length > 0
      ? turnsWithDelivery
          .map((t, i) => {
            const d = t.deliveryAnalysis;
            return `Q${i + 1}: Delivery ${d.deliveryScore}/100, Fillers: ${d.fillerCount}, Hedging: ${d.hedgingCount}, Restarts: ${d.sentenceRestarts}. Top communication improvement: ${d.topImprovement}. Top communication strength: ${d.topStrength}`;
          })
          .join("\n")
      : "No delivery analysis available.";

  const historyText = history
    .map(
      (h, i) =>
        `Q${i + 1} [${h.difficulty || "Medium"}]: ${h.question}\nTechnical Score: ${h.score}/100`,
    )
    .join("\n\n");

  let interviewContextLine;
  if (interviewType === "TECHNICAL" && config) {
    interviewContextLine = `INTERVIEW TYPE: Technical — ${config.stack || "General"} (${config.difficulty || "Medium"} level, ${config.questionCount || history.length} questions)`;
  } else if (interviewType === "BEHAVIORAL" && config) {
    interviewContextLine = `INTERVIEW TYPE: Behavioral (STAR method, ${config.difficulty || "Medium"} level, ${config.questionCount || history.length} questions)`;
  } else {
    interviewContextLine = `INTERVIEW TYPE: Job-Specific\n    - Resume Gaps: ${JSON.stringify(gapAnalysis)}`;
  }
  const videoAnalyses = prefetchedVideoData || [];
  const hasVideoData = videoAnalyses.length > 0;
  const videoScore = hasVideoData
    ? videoAnalyses.reduce((sum, v) => sum + v.confidenceLevel * 100, 0) /
      videoAnalyses.length
    : null;
  const fusedConfidenceValues = history
    .map((turn) => {
      const S_audio = turn.voiceAnalysis?.confidenceLevel ?? null;
      const S_video = turn.videoAnalysis?.confidenceLevel ?? null;

      if (S_audio !== null && S_video !== null) {
        return W_AUDIO * S_audio + W_VIDEO * S_video;
      }
      if (S_audio !== null) return S_audio;
      if (S_video !== null) return S_video;
      return null;
    })
    .filter((score) => score !== null);
  const fusedScore =
    fusedConfidenceValues.length > 0
      ? parseFloat(
          (
            (fusedConfidenceValues.reduce((sum, score) => sum + score, 0) /
              fusedConfidenceValues.length) *
            100
          ).toFixed(2),
        )
      : null;

  const prompt = `
    You are an expert Technical Hiring Manager.

    ${interviewContextLine}

    TECHNICAL DATA:
    - Score: ${technicalScore.toFixed(1)}/100 (difficulty-adjusted)
    - History: ${historyText}
    - Note: The interview used adaptive difficulty — harder questions were asked when the candidate performed well. A lower score on a Hard question is more acceptable than the same score on an Easy question, especially for junior/entry-level roles.

    COMMUNICATION DATA:
    - Vocal Confidence Score: ${voiceScore !== null ? voiceScore.toFixed(1) + "/100" : "N/A (chat-only interview)"}
    - Vocal Observations: ${allVocalInsights.length > 0 ? allVocalInsights.join("; ") : hasVoiceData ? "Stable and confident delivery." : "No audio data — chat-only interview."}

    DELIVERY ANALYSIS (transcript quality):
    - Overall Delivery Score: ${deliveryScore !== null ? deliveryScore.toFixed(1) + "/100" : "N/A"}
    - Per-question breakdown:
    ${deliveryInsights}

    BODY LANGUAGE DATA:
    - Video Confidence Score: ${videoScore !== null ? videoScore.toFixed(1) + "/100" : "N/A (no video data)"}

    TASK:
    Evaluate the overall candidate considering technical knowledge, vocal delivery, AND answer quality/structure.
    Look for "Confidence Mismatches": If the technical score is high but vocal insights mention
    "vocal tremors" or "frequent hesitations," note that they may know the theory but lack
    confidence in explaining it under pressure.
    Also note delivery patterns: excessive filler words, hedging language, or lack of specificity.
    DIFFICULTY CURVE: If the candidate answered Easy/Medium questions well but struggled on Hard stretch questions, treat this as a POSITIVE signal for junior/mid roles — they demonstrated competence at their level and the system pushed them to their ceiling. Only treat Hard question struggles negatively for Senior roles where Hard is the expected baseline.
  `;

  const result = await generateStructured(prompt, FinalReportSchema);

  // 4b. Aggregate Video Score

  // 5. Final Combined Payload for the Database & UI
  const contentQuality =
    deliveryScore !== null
      ? technicalScore * 0.6 + deliveryScore * 0.4
      : technicalScore;

  let combined;
  if (voiceScore !== null && videoScore !== null) {
    combined = contentQuality * 0.41 + videoScore * 0.32 + voiceScore * 0.27;
  } else if (voiceScore !== null) {
    combined = contentQuality * 0.6 + voiceScore * 0.4;
  } else if (videoScore !== null) {
    combined = contentQuality * 0.56 + videoScore * 0.44;
  } else {
    combined = contentQuality;
  }

  const deliverySummary =
    turnsWithDelivery.length > 0
      ? {
          avgDeliveryScore: parseFloat(deliveryScore.toFixed(1)),
          totalFillers: turnsWithDelivery.reduce(
            (s, t) => s + t.deliveryAnalysis.fillerCount,
            0,
          ),
          totalHedging: turnsWithDelivery.reduce(
            (s, t) => s + t.deliveryAnalysis.hedgingCount,
            0,
          ),
          totalRestarts: turnsWithDelivery.reduce(
            (s, t) => s + t.deliveryAnalysis.sentenceRestarts,
            0,
          ),
        }
      : null;

  return {
    ...result,
    scores: {
      technical: parseFloat(technicalScore.toFixed(1)),
      voice: voiceScore !== null ? parseFloat(voiceScore.toFixed(1)) : null,
      video: videoScore !== null ? parseFloat(videoScore.toFixed(1)) : null,
      fusedScore,
      delivery:
        deliveryScore !== null ? parseFloat(deliveryScore.toFixed(1)) : null,
      contentQuality: parseFloat(contentQuality.toFixed(1)),
      combined: parseFloat(combined.toFixed(1)),
    },
    voiceSummary: hasVoiceData
      ? {
          overallLabel:
            voiceScore > 75
              ? "Highly Confident"
              : voiceScore > 50
                ? "Moderately Confident"
                : "Needs Improvement",
          allInsights: allVocalInsights,
          avgWPM: (
            voiceAnalyses.reduce((sum, v) => sum + v.wordsPerMinute, 0) /
            voiceAnalyses.length
          ).toFixed(0),
        }
      : null,
    deliverySummary,
  };
}
