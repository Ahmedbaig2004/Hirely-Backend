import { prisma } from "../config/db.js";

// GET /api/coding-questions?difficulty=Easy&category=Two+Pointers&topic=Array&page=1&limit=20
export const getCodingQuestions = async (req, res) => {
  try {
    const { difficulty, category, topic, page = 1, limit = 20 } = req.query;
    const skip = (parseInt(page) - 1) * parseInt(limit);

    const where = {};
    if (difficulty) where.difficulty = difficulty;
    if (category) where.category = category;
    if (topic) where.topics = { has: topic };

    const [questions, total] = await Promise.all([
      prisma.codingQuestion.findMany({
        where,
        select: {
          id: true,
          problemId: true,
          title: true,
          slug: true,
          difficulty: true,
          category: true,
          topics: true,
        },
        orderBy: { problemId: "asc" },
        skip,
        take: parseInt(limit),
      }),
      prisma.codingQuestion.count({ where }),
    ]);

    res.json({ questions, total, page: parseInt(page), limit: parseInt(limit) });
  } catch (error) {
    console.error("getCodingQuestions error:", error);
    res.status(500).json({ error: "Failed to fetch coding questions" });
  }
};

// GET /api/coding-questions/:id  (id = LeetCode problemId number)
export const getCodingQuestionById = async (req, res) => {
  try {
    const problemId = parseInt(req.params.id, 10);
    if (isNaN(problemId)) return res.status(400).json({ error: "Invalid problem ID" });

    const question = await prisma.codingQuestion.findUnique({
      where: { problemId },
    });

    if (!question) return res.status(404).json({ error: "Question not found" });

    res.json(question);
  } catch (error) {
    console.error("getCodingQuestionById error:", error);
    res.status(500).json({ error: "Failed to fetch coding question" });
  }
};
