-- CreateSchema
CREATE SCHEMA IF NOT EXISTS "public";

-- CreateExtension
CREATE EXTENSION IF NOT EXISTS "vector";

-- CreateTable
CREATE TABLE "Interview" (
    "id" TEXT NOT NULL,
    "interviewType" TEXT NOT NULL DEFAULT 'JOB_SPECIFIC',
    "config" JSONB,
    "jobDescription" TEXT,
    "userId" TEXT NOT NULL,
    "finalScore" DOUBLE PRECISION,
    "finalFeedback" JSONB,
    "status" TEXT NOT NULL DEFAULT 'processing',
    "errorLog" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Interview_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "InterviewTurn" (
    "id" SERIAL NOT NULL,
    "interviewId" TEXT NOT NULL,
    "question" TEXT NOT NULL,
    "answer" TEXT NOT NULL,
    "score" INTEGER NOT NULL,
    "feedback" TEXT NOT NULL,
    "improvedAnswer" TEXT,
    "topic" TEXT,
    "difficulty" TEXT,
    "softSkillScore" INTEGER,
    "answerMode" TEXT,
    "audioUrl" TEXT,
    "deliveryScore" DOUBLE PRECISION,
    "fillerCount" INTEGER,
    "hedgingCount" INTEGER,
    "sentenceRestarts" INTEGER,
    "relevanceScore" DOUBLE PRECISION,
    "specificityScore" DOUBLE PRECISION,
    "deliveryFeedback" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "InterviewTurn_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "VoiceAnalysis" (
    "id" SERIAL NOT NULL,
    "interviewTurnId" INTEGER NOT NULL,
    "confidenceLevel" DOUBLE PRECISION,
    "confidenceLabelText" TEXT,
    "speakingQuality" DOUBLE PRECISION,
    "vocalStability" DOUBLE PRECISION,
    "speakingFluency" DOUBLE PRECISION,
    "pitchMean" DOUBLE PRECISION,
    "pitchStd" DOUBLE PRECISION,
    "energyLevel" DOUBLE PRECISION,
    "wordsPerMinute" DOUBLE PRECISION,
    "pauseRatio" DOUBLE PRECISION,
    "jitter" DOUBLE PRECISION,
    "shimmer" DOUBLE PRECISION,
    "modelVersion" TEXT,
    "allProbabilities" JSONB,
    "rawFeatures" JSONB,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "errorMessage" TEXT,
    "processingTimeMs" INTEGER,
    "processedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "VoiceAnalysis_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "VideoAnalysis" (
    "id" SERIAL NOT NULL,
    "interviewTurnId" INTEGER NOT NULL,
    "confidenceLevel" DOUBLE PRECISION,
    "confidenceLabelText" TEXT,
    "rawScore" DOUBLE PRECISION,
    "modelVersion" TEXT,
    "rawFeatures" JSONB,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "errorMessage" TEXT,
    "processingTimeMs" INTEGER,
    "processedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "VideoAnalysis_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Document" (
    "id" SERIAL NOT NULL,
    "content" TEXT NOT NULL,
    "embedding" vector(3072),
    "metadata" JSONB,
    "source" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Document_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "CodingQuestion" (
    "id" SERIAL NOT NULL,
    "problemId" INTEGER NOT NULL,
    "title" TEXT NOT NULL,
    "slug" TEXT NOT NULL,
    "difficulty" TEXT NOT NULL,
    "category" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "topics" TEXT[],
    "examples" JSONB,
    "constraints" TEXT[],
    "codeSnippets" JSONB,
    "hints" TEXT[],
    "solution" TEXT,
    "masterSolution" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "CodingQuestion_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "TestCase" (
    "id" SERIAL NOT NULL,
    "codingQuestionId" INTEGER NOT NULL,
    "input" TEXT NOT NULL,
    "expectedOutput" TEXT NOT NULL,
    "category" TEXT NOT NULL,
    "isVerified" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "TestCase_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "Interview_userId_idx" ON "Interview"("userId");

-- CreateIndex
CREATE INDEX "Interview_createdAt_idx" ON "Interview"("createdAt");

-- CreateIndex
CREATE INDEX "InterviewTurn_interviewId_idx" ON "InterviewTurn"("interviewId");

-- CreateIndex
CREATE UNIQUE INDEX "VoiceAnalysis_interviewTurnId_key" ON "VoiceAnalysis"("interviewTurnId");

-- CreateIndex
CREATE UNIQUE INDEX "VideoAnalysis_interviewTurnId_key" ON "VideoAnalysis"("interviewTurnId");

-- CreateIndex
CREATE UNIQUE INDEX "CodingQuestion_problemId_key" ON "CodingQuestion"("problemId");

-- CreateIndex
CREATE UNIQUE INDEX "CodingQuestion_title_key" ON "CodingQuestion"("title");

-- CreateIndex
CREATE UNIQUE INDEX "CodingQuestion_slug_key" ON "CodingQuestion"("slug");

-- CreateIndex
CREATE INDEX "TestCase_codingQuestionId_idx" ON "TestCase"("codingQuestionId");

-- AddForeignKey
ALTER TABLE "InterviewTurn" ADD CONSTRAINT "InterviewTurn_interviewId_fkey" FOREIGN KEY ("interviewId") REFERENCES "Interview"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "VoiceAnalysis" ADD CONSTRAINT "VoiceAnalysis_interviewTurnId_fkey" FOREIGN KEY ("interviewTurnId") REFERENCES "InterviewTurn"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "VideoAnalysis" ADD CONSTRAINT "VideoAnalysis_interviewTurnId_fkey" FOREIGN KEY ("interviewTurnId") REFERENCES "InterviewTurn"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TestCase" ADD CONSTRAINT "TestCase_codingQuestionId_fkey" FOREIGN KEY ("codingQuestionId") REFERENCES "CodingQuestion"("id") ON DELETE CASCADE ON UPDATE CASCADE;