import {
  generateInterviewContext,
  generateTechnicalQuestions,
  generateBehavioralQuestions,
} from "./analyzer.js";

/**
 * Factory that dispatches initial question generation based on interview type.
 *
 * Returns a normalized shape:
 *   { candidateSummary, gapAnalysis, questions[] }
 *
 * candidateSummary and gapAnalysis are null for Technical / Behavioral types
 * so downstream code doesn't need to branch.
 */
export async function generateInitialQuestions({
  interviewType,
  resumeBuffer,
  jobDescription,
  config,
  interviewMode,
}) {
  if (interviewType === "JOB_SPECIFIC") {
    // Generate only the 2 opening questions we actually use before adaptive mode.
    const analysis = await generateInterviewContext(
      resumeBuffer,
      jobDescription,
      2,
      interviewMode,
    );
    return {
      candidateSummary: analysis.candidateSummary,
      gapAnalysis: analysis.gapAnalysis,
      questions: analysis.questions,
    };
  }

  if (interviewType === "TECHNICAL") {
    const { stack, difficulty, questionCount } = config;
    const seedCount = Math.min(questionCount, 2);
    const questions = await generateTechnicalQuestions(
      stack,
      difficulty,
      seedCount,
      interviewMode,
    );
    return { candidateSummary: null, gapAnalysis: null, questions };
  }

  if (interviewType === "BEHAVIORAL") {
    const { difficulty, questionCount } = config;
    const seedCount = Math.min(questionCount, 2);
    const questions = await generateBehavioralQuestions(
      difficulty,
      seedCount,
      interviewMode,
    );
    return { candidateSummary: null, gapAnalysis: null, questions };
  }

  throw new Error(`Unknown interviewType: ${interviewType}`);
}
