import { PrismaClient } from "../generated/prisma/index.js";
const prisma = new PrismaClient();

// GET /api/interviews
export const getInterviews = async (req, res) => {
  try {
    const interviews = await prisma.interview.findMany({
      orderBy: { createdAt: "desc" },
      select: {
        id: true,
        jobDescription: true,
        finalScore: true,
        createdAt: true,
        // We don't need the full 'turns' or 'feedback' for the list view
      },
    });
    res.json(interviews);
  } catch (error) {
    console.error("Dashboard Error:", error);
    res.status(500).json({ error: "Failed to fetch interviews" });
  }
};

// GET /api/interviews/:id
export const getInterviewDetail = async (req, res) => {
  const { id } = req.params;
  try {
    const interview = await prisma.interview.findUnique({
      where: { id },
      include: { turns: true }, // Get the Q&A history
    });
    if (!interview) return res.status(404).json({ error: "Not found" });

    res.json(interview);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch detail" });
  }
};
