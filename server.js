import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import rateLimit from "express-rate-limit";
import healthRoutes from "./routes/healthRoutes.js";
import interviewRoutes from "./routes/interviewRoutes.js";
import evaluateRoutes from "./routes/evaluateRoutes.js";
import dashboardRoutes from "./routes/dashboardRoutes.js";
import codingQuestionRoutes from "./routes/codingQuestionRoutes.js";
dotenv.config();

const app = express();
const PORT = process.env.PORT || 4000;

// --- 1. CONFIGURATION ---

// A. CORS (Allow frontend to connect)
app.use(
  cors({
    origin: process.env.FRONTEND_URL || "http://localhost:3000",
    methods: ["GET", "POST", "DELETE"],
  }),
);

app.use(express.json());

// B. GENERAL LIMITER (Anti-Spam / DDoS Protection)
// Local dev does a lot of hot reload/fetch retries, so keep production strict
// while avoiding accidental dashboard lockouts during development.
const isProduction = process.env.NODE_ENV === "production";
const rateLimitMax = Number(
  process.env.RATE_LIMIT_MAX ?? (isProduction ? 1000 : 1000),
);
const generalLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: rateLimitMax,
  message: "Too many requests from this IP, please try again after 15 minutes",
  standardHeaders: true,
  legacyHeaders: false,
});

// Apply General Limiter to ALL routes
app.use(generalLimiter);

// --- 2. ROUTES ---

// Health Check
app.use("/", healthRoutes);

// API Routes
app.use("/api", interviewRoutes);
app.use("/api", evaluateRoutes);
app.use("/api", dashboardRoutes);
app.use("/api", codingQuestionRoutes);
app.listen(PORT, () => {
  console.log(`\n🚀 HIRELY Backend running on http://localhost:${PORT}`);
});
