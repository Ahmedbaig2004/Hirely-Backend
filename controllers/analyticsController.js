import { prisma } from "../config/db.js";

/** Null-safe average: ignores null/NaN values. Returns null if no valid values. */
function avg(values) {
  const valid = values.filter((v) => v != null && !Number.isNaN(v));
  if (valid.length === 0) return null;
  return valid.reduce((s, v) => s + v, 0) / valid.length;
}

/** Get Monday of the ISO week for a given date */
function isoWeekStart(date) {
  const d = new Date(date);
  const day = d.getUTCDay(); // 0 = Sun, 1 = Mon …
  const diff = (day === 0 ? -6 : 1 - day);
  d.setUTCDate(d.getUTCDate() + diff);
  d.setUTCHours(0, 0, 0, 0);
  return d.toISOString().slice(0, 10); // "YYYY-MM-DD"
}

/** Get "YYYY-MM" for a given date */
function monthBucket(date) {
  return new Date(date).toISOString().slice(0, 7);
}

function normalizeDifficulty(difficulty) {
  const value = String(difficulty || "Medium").trim().toLowerCase();
  if (value === "easy") return "Easy";
  if (value === "hard") return "Hard";
  return "Medium";
}

function toPct(value) {
  if (value == null || Number.isNaN(value)) return null;
  return value <= 1 ? value * 100 : value;
}

function collectTopVideoSignals(videoAnalyses) {
  const byName = {};
  for (const video of videoAnalyses) {
    const groups = video.rawFeatures;
    if (!groups || typeof groups !== "object") continue;

    for (const group of Object.values(groups)) {
      if (!group || typeof group !== "object") continue;
      const tips = Array.isArray(group.tips) ? group.tips : [];
      for (const tip of tips) {
        const name = tip.friendly || tip.feature;
        const impact = typeof tip.shap === "number" ? tip.shap : 0;
        if (!name || impact === 0) continue;
        byName[name] = (byName[name] || 0) + impact;
      }
    }
  }

  return Object.entries(byName)
    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
    .slice(0, 5)
    .map(([name, impact]) => ({ name, impact: Math.round(impact * 10) / 10 }));
}

// GET /api/analytics
export const getAnalytics = async (req, res) => {
  try {
    const { userId, type, from, to } = req.query;

    if (!userId) {
      return res.status(400).json({ error: "userId is required" });
    }

    // --- Build date filter ---
    const dateFilter = {};
    if (from) dateFilter.gte = new Date(from);
    if (to) {
      const toDate = new Date(to);
      toDate.setUTCHours(23, 59, 59, 999);
      dateFilter.lte = toDate;
    }

    // --- Query ---
    const interviews = await prisma.interview.findMany({
      where: {
        userId,
        ...(type && type !== "all" && { interviewType: type }),
        ...(Object.keys(dateFilter).length > 0 && { createdAt: dateFilter }),
      },
      orderBy: { createdAt: "asc" },
      select: {
        id: true,
        interviewType: true,
        finalScore: true,
        finalFeedback: true,
        createdAt: true,
        config: true,
        turns: {
          select: {
            score: true,
            difficulty: true,
            topic: true,
            deliveryScore: true,
            fillerCount: true,
            hedgingCount: true,
            sentenceRestarts: true,
            voiceAnalysis: {
              select: {
                confidenceLevel: true,
                wordsPerMinute: true,
                pauseRatio: true,
                speakingFluency: true,
                vocalStability: true,
                speakingQuality: true,
              },
            },
            videoAnalysis: {
              select: {
                confidenceLevel: true,
                confidenceLabelText: true,
                rawScore: true,
                rawFeatures: true,
                status: true,
              },
            },
          },
        },
      },
    });

    if (interviews.length === 0) {
      return res.json({
        kpis: {
          totalInterviews: 0,
          avgFinalScore: null,
          bestScore: null,
          totalQuestionsAnswered: 0,
          hireRate: null,
        },
        trend: [],
        byType: [],
        modality: { technical: null, delivery: null, voice: null, video: null, contentQuality: null, combined: null },
        topicHeatmap: [],
        delivery: { avgFillers: null, avgHedging: null, avgRestarts: null },
        video: { avgConfidence: null, avgRawScore: null, analyzedTurns: 0, labelBreakdown: [], topSignals: [] },
        decisionBreakdown: [],
      });
    }

    // --- All turns flat ---
    const allTurns = interviews.flatMap((iv) => iv.turns);

    // -----------------------------------------------------------------------
    // KPIs
    // -----------------------------------------------------------------------
    const finalScores = interviews.map((iv) => iv.finalScore).filter((s) => s != null);
    const hires = interviews.filter((iv) =>
      ["Strong Hire", "Hire"].includes(iv.finalFeedback?.decision)
    ).length;

    const kpis = {
      totalInterviews: interviews.length,
      avgFinalScore: avg(finalScores) !== null ? Math.round(avg(finalScores)) : null,
      bestScore: finalScores.length > 0 ? Math.round(Math.max(...finalScores)) : null,
      totalQuestionsAnswered: allTurns.length,
      hireRate: Math.round((hires * 100) / interviews.length),
    };

    // -----------------------------------------------------------------------
    // Score Trend — weekly if ≤90 days, else monthly
    // -----------------------------------------------------------------------
    const rangeMs =
      interviews.length > 0
        ? new Date(interviews[interviews.length - 1].createdAt) - new Date(interviews[0].createdAt)
        : 0;
    const useWeekly = rangeMs <= 90 * 24 * 60 * 60 * 1000;

    const trendBuckets = {};
    for (const iv of interviews) {
      const bucket = useWeekly ? isoWeekStart(iv.createdAt) : monthBucket(iv.createdAt);
      if (!trendBuckets[bucket]) trendBuckets[bucket] = { scores: [], count: 0 };
      trendBuckets[bucket].count += 1;
      if (iv.finalScore != null) trendBuckets[bucket].scores.push(iv.finalScore);
    }
    const trend = Object.entries(trendBuckets)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, { scores, count }]) => ({
        date,
        avgScore: avg(scores) !== null ? Math.round(avg(scores)) : null,
        count,
      }));

    // -----------------------------------------------------------------------
    // By Interview Type
    // -----------------------------------------------------------------------
    const TYPE_LIST = ["JOB_SPECIFIC", "TECHNICAL", "BEHAVIORAL"];
    const byType = TYPE_LIST.map((t) => {
      const group = interviews.filter((iv) => iv.interviewType === t);
      if (group.length === 0) return null;
      const groupTurns = group.flatMap((iv) => iv.turns);
      return {
        type: t,
        count: group.length,
        avgScore: avg(group.map((iv) => iv.finalScore).filter((s) => s != null)) !== null
          ? Math.round(avg(group.map((iv) => iv.finalScore).filter((s) => s != null)))
          : null,
        avgDelivery: avg(groupTurns.map((t) => t.deliveryScore).filter((s) => s != null)) !== null
          ? Math.round(avg(groupTurns.map((t) => t.deliveryScore).filter((s) => s != null)))
          : null,
        avgVoice: avg(groupTurns.map((t) => t.voiceAnalysis?.confidenceLevel).filter((s) => s != null)) !== null
          ? Math.round(avg(groupTurns.map((t) => t.voiceAnalysis?.confidenceLevel).filter((s) => s != null)) * 100)
          : null,
        avgVideo: avg(groupTurns.map((t) => t.videoAnalysis?.confidenceLevel).filter((s) => s != null)) !== null
          ? Math.round(avg(groupTurns.map((t) => t.videoAnalysis?.confidenceLevel).filter((s) => s != null)) * 100)
          : null,
      };
    }).filter(Boolean);

    // -----------------------------------------------------------------------
    // Modality Averages (across all matching interviews, from finalFeedback.scores)
    // -----------------------------------------------------------------------
    const fb = interviews.map((iv) => iv.finalFeedback?.scores).filter(Boolean);
    const modality = {
      technical: avg(fb.map((s) => s.technical).filter((v) => v != null)) !== null
        ? Math.round(avg(fb.map((s) => s.technical).filter((v) => v != null)))
        : null,
      delivery: avg(fb.map((s) => s.delivery).filter((v) => v != null)) !== null
        ? Math.round(avg(fb.map((s) => s.delivery).filter((v) => v != null)))
        : null,
      voice: avg(fb.map((s) => s.voice).filter((v) => v != null)) !== null
        ? Math.round(avg(fb.map((s) => s.voice).filter((v) => v != null)))
        : null,
      video: avg(fb.map((s) => s.video).filter((v) => v != null)) !== null
        ? Math.round(avg(fb.map((s) => s.video).filter((v) => v != null)))
        : null,
      contentQuality: avg(fb.map((s) => s.contentQuality).filter((v) => v != null)) !== null
        ? Math.round(avg(fb.map((s) => s.contentQuality).filter((v) => v != null)))
        : null,
      combined: avg(fb.map((s) => s.combined).filter((v) => v != null)) !== null
        ? Math.round(avg(fb.map((s) => s.combined).filter((v) => v != null)))
        : null,
    };

    // -----------------------------------------------------------------------
    // Topic Heatmap — top 10 topics by turn count
    // -----------------------------------------------------------------------
    const topicMap = {};
    for (const turn of allTurns) {
      const topic = turn.topic || "General";
      const difficulty = normalizeDifficulty(turn.difficulty);
      if (!topicMap[topic]) topicMap[topic] = { Easy: [], Medium: [], Hard: [], count: 0 };
      topicMap[topic].count += 1;
      if (turn.score != null && !Number.isNaN(turn.score)) {
        topicMap[topic][difficulty].push(turn.score);
      }
    }

    const topicHeatmap = Object.entries(topicMap)
      .sort(([, a], [, b]) => b.count - a.count)
      .slice(0, 10)
      .map(([topic, data]) => ({
        topic,
        count: data.count,
        Easy: data.Easy.length > 0 ? Math.round(avg(data.Easy)) : null,
        Medium: data.Medium.length > 0 ? Math.round(avg(data.Medium)) : null,
        Hard: data.Hard.length > 0 ? Math.round(avg(data.Hard)) : null,
      }));

    // -----------------------------------------------------------------------
    // Delivery Aggregates
    // -----------------------------------------------------------------------
    const turnsWithDelivery = allTurns.filter((t) => t.deliveryScore != null);
    const delivery = {
      avgFillers: avg(turnsWithDelivery.map((t) => t.fillerCount)) !== null
        ? Math.round(avg(turnsWithDelivery.map((t) => t.fillerCount)) * 10) / 10
        : null,
      avgHedging: avg(turnsWithDelivery.map((t) => t.hedgingCount)) !== null
        ? Math.round(avg(turnsWithDelivery.map((t) => t.hedgingCount)) * 10) / 10
        : null,
      avgRestarts: avg(turnsWithDelivery.map((t) => t.sentenceRestarts)) !== null
        ? Math.round(avg(turnsWithDelivery.map((t) => t.sentenceRestarts)) * 10) / 10
        : null,
    };

    // -----------------------------------------------------------------------
    // Video Presence Aggregates
    // -----------------------------------------------------------------------
    const turnsWithVideo = allTurns.filter(
      (t) =>
        t.videoAnalysis &&
        (t.videoAnalysis.status === "completed" ||
          t.videoAnalysis.confidenceLevel != null)
    );
    const videoAnalyses = turnsWithVideo.map((t) => t.videoAnalysis);
    const videoLabelCounts = {};
    for (const video of videoAnalyses) {
      const label = video.confidenceLabelText || "Needs work";
      videoLabelCounts[label] = (videoLabelCounts[label] || 0) + 1;
    }
    const video = {
      avgConfidence:
        avg(videoAnalyses.map((v) => toPct(v.confidenceLevel)).filter((v) => v != null)) !== null
          ? Math.round(avg(videoAnalyses.map((v) => toPct(v.confidenceLevel)).filter((v) => v != null)))
          : null,
      avgRawScore:
        avg(videoAnalyses.map((v) => toPct(v.rawScore)).filter((v) => v != null)) !== null
          ? Math.round(avg(videoAnalyses.map((v) => toPct(v.rawScore)).filter((v) => v != null)))
          : null,
      analyzedTurns: videoAnalyses.length,
      labelBreakdown: Object.entries(videoLabelCounts).map(([label, count]) => ({
        label,
        count,
      })),
      topSignals: collectTopVideoSignals(videoAnalyses),
    };

    // -----------------------------------------------------------------------
    // Decision Breakdown
    // -----------------------------------------------------------------------
    const DECISIONS = ["Strong Hire", "Hire", "Weak Hire", "No Hire"];
    const decisionCounts = {};
    for (const iv of interviews) {
      const d = iv.finalFeedback?.decision;
      if (d) decisionCounts[d] = (decisionCounts[d] || 0) + 1;
    }
    const decisionBreakdown = DECISIONS.filter((d) => decisionCounts[d] > 0).map((decision) => ({
      decision,
      count: decisionCounts[decision],
    }));

    return res.json({ kpis, trend, byType, modality, topicHeatmap, delivery, video, decisionBreakdown });
  } catch (error) {
    console.error("Analytics Error:", error);
    res.status(500).json({ error: "Failed to fetch analytics" });
  }
};
