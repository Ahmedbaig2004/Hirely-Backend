import { prisma } from "../config/db.js";

// GET /api/interviews
export const getInterviews = async (req, res) => {
  try {
    const { userId } = req.query; // Get from query param
    const interviews = await prisma.interview.findMany({
      orderBy: { createdAt: "desc" },
      where: { userId: userId }, // <--- FILTER HERE
      select: {
        id: true,
        interviewType: true,
        jobDescription: true,
        finalScore: true,
        finalFeedback: true,
        createdAt: true,
      },
    });
    res.json(interviews);
  } catch (error) {
    console.error("Dashboard Error:", error);
    res.status(500).json({ error: "Failed to fetch interviews" });
  }
};

// DELETE /api/interviews/:id
export const deleteInterview = async (req, res) => {
  const { id } = req.params;
  const { userId } = req.query;

  try {
    const interview = await prisma.interview.findUnique({ where: { id } });
    if (!interview) return res.status(404).json({ error: "Not found" });
    if (interview.userId !== userId) {
      return res.status(403).json({ error: "Unauthorized: Cannot delete this interview" });
    }
    await prisma.interview.delete({ where: { id } });
    res.json({ success: true });
  } catch (error) {
    console.error("Delete Error:", error);
    res.status(500).json({ error: "Failed to delete interview" });
  }
};

// GET /api/interviews/:id
export const getInterviewDetail = async (req, res) => {
  const { id } = req.params;
  const { userId } = req.query; // Get userId from request (should be from auth)

  try {
    const interview = await prisma.interview.findUnique({
      where: { id },
      include: {
        turns: {
          include: { voiceAnalysis: true },
          orderBy: { createdAt: "asc" },
        },
      },
    });

    if (!interview) return res.status(404).json({ error: "Not found" });

    // ✅ CRITICAL: Verify the user owns this interview
    if (interview.userId !== userId) {
      return res.status(403).json({ error: "Unauthorized: Cannot access this interview" });
    }

    res.json(interview);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch detail" });
  }
};
