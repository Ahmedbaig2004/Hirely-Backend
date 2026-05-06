import { generateStructured } from "../config/gemini.js";
import { z } from "zod";
import dotenv from "dotenv";

dotenv.config();

const DeliveryAnalysisSchema = z.object({
  deliveryScore: z
    .number()
    .min(0)
    .max(100)
    .describe(
      "Overall delivery quality score (0-100). Considers structure, clarity, filler usage, hedging, and sentence flow.",
    ),
  fillerCount: z
    .number()
    .describe(
      "Total number of filler words detected (um, uh, like, you know, basically, literally, sort of, kind of, I mean)",
    ),
  fillerWords: z
    .array(
      z.object({
        word: z.string(),
        count: z.number(),
      }),
    )
    .describe("Breakdown of each filler word and how many times it appeared"),
  structureFeedback: z
    .string()
    .describe(
      "One sentence assessing answer structure: did it have a clear opening, supporting detail, and conclusion?",
    ),
  hedgingCount: z
    .number()
    .describe(
      'Number of hedging/uncertain phrases ("I think maybe", "probably", "sort of", "I guess", "not sure but")',
    ),
  hedgingPhrases: z
    .array(z.string())
    .describe("The actual hedging phrases detected in the transcript"),
  sentenceRestarts: z
    .number()
    .describe(
      "Number of times the speaker started a sentence, stopped, and restarted",
    ),
  topImprovement: z
    .string()
    .describe(
      "The single most impactful communication improvement. Keep it actionable and focused on delivery, not technical content.",
    ),
  topStrength: z
    .string()
    .describe(
      "The single strongest communication aspect of this answer's delivery.",
    ),
});

/**
 * Analyze transcript delivery quality using Gemini structured output.
 * Returns actionable feedback on fillers, hedging, structure, and flow.
 *
 * @param {string} transcript - The candidate's answer text
 * @param {string} question - The interview question that was asked
 * @returns {Promise<object>} Structured delivery analysis
 */
export async function analyzeDelivery(transcript, question, language = "en") {
  if (!transcript || transcript.trim().length < 10) {
    return null;
  }

  const fillerInstructions = language === "ur"
    ? `- The answer is in Roman Urdu (Urdu written in Latin script, mixed with English technical terms). Evaluate structure and relevance based on content meaning, not language choice.
- Count these Urdu filler words: yani, matlab, woh (when used as filler), aisa, haan, bilkul (when used as filler not genuine emphasis).
- Also count any English fillers that appear: um, uh, like (non-comparison), you know, basically, literally, sort of, kind of, I mean.`
    : `- Count filler words carefully. Common fillers: um, uh, like (when not used as comparison), you know, basically, literally, sort of, kind of, I mean.`;

  const prompt = `You are an expert interview communication coach analyzing a candidate's answer delivery.

INTERVIEW QUESTION:
"${question}"

CANDIDATE'S ANSWER (transcribed from speech):
"${transcript}"

TASK: Analyze the DELIVERY quality of this answer — not whether it's technically correct, but HOW it was communicated.

Scoring guide:
- deliveryScore: 85-100 = excellent (clear, structured, concise, minimal fillers). 70-84 = good (minor communication issues). 50-69 = needs work (noticeable fillers, hedging, or weak structure). Below 50 = significant delivery problems.
- Focus scoring on communication only, not technical depth or completeness.

Important:
${fillerInstructions}
- Hedging phrases indicate lack of confidence: "I think maybe", "probably", "not sure but", "I guess", "sort of like".
- Sentence restarts: "I worked on — well actually I was responsible for —" counts as 1 restart.
- Be precise with counts — do not overcount or undercount.
- Focus only on communication delivery: structure, fillers, hedging, pauses, clarity, directness, and sentence flow.
- Do not critique technical correctness, missing technical depth, examples, trade-offs, or content completeness.
- The topImprovement should be something the candidate can practice and fix before their next interview.
- The topStrength should also describe communication only, not technical knowledge.`;

  try {
    return await generateStructured(prompt, DeliveryAnalysisSchema);
  } catch (e) {
    console.error("Delivery analysis failed:", e.message);
    return null;
  }
}
