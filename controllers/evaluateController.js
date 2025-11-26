import { evaluateAnswer } from "../services/retrieval.js";

/**
 * Manual evaluation endpoint (Text only)
 * POST /api/evaluate
 */
export const evaluate = async (req, res) => {
  try {
    const { question, answer } = req.body;
    const result = await evaluateAnswer(question, answer);
    res.json(result);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "Failed to evaluate answer" });
  }
};

