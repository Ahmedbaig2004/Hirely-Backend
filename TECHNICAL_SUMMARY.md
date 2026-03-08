# HIRELY Backend - Complete Technical Summary

## 📋 Project Overview
**Hirely** is an AI-powered technical interview platform that conducts automated video/audio interviews with candidates. The system analyzes resumes, generates tailored technical questions, evaluates answers in real-time, and produces comprehensive hiring recommendations.

---

## 🎯 Core Purpose
- **Resume Analysis**: Parse PDF resumes and extract candidate information
- **Gap Analysis**: Identify missing skills/qualifications compared to job description
- **Dynamic Interview Generation**: Create contextual technical questions based on resume-JD match
- **Real-time Answer Evaluation**: Grade candidate responses using semantic similarity + LLM judgment
- **Interview State Management**: Track interview sessions and history across distributed requests
- **Final Hiring Report**: Generate structured recommendations (Strong Hire/Hire/Weak Hire/No Hire)

---

## 🏗️ Architecture

### Tech Stack
**Runtime & Framework:**
- Node.js (ESM modules)
- Express.js v5.1.0 (REST API)

**AI/ML Services:**
- Google Gemini API (gemini-2.5-flash, gemini-2.5-flash-lite)
- LangChain 1.0.1 (@langchain/core, @langchain/community, @langchain/google-genai)
- Text embeddings (text-embedding-004 model)
- Xenova Transformers (2.17.2) for local NLP

**Database & Storage:**
- PostgreSQL + Prisma ORM (v6.16.2)
- PostgreSQL Vector Extension (pgvector) for semantic search
- Redis 5.8.2 (session state management)
- Supabase 2.57.4 (authentication, storage)

**Audio/Media Processing:**
- Deepgram SDK 4.11.2 (speech-to-text transcription)
- FFmpeg 5.3.0 (audio processing)
- Fluent-FFmpeg 2.1.3 (wrapper)
- WaveFile 11.0.0 (audio manipulation)

**Data Processing:**
- PDF-Parse 1.1.1 (resume parsing)
- Cheerio 1.1.2 (HTML parsing)
- Zod 3.23.8 (schema validation)
- Cosine-Similarity (vector similarity computation)

**Utilities:**
- CORS 2.8.5
- Express Rate Limit 8.2.1 (anti-spam: 100 req/15 min)
- Multer 2.0.2 (file uploads)
- Groq SDK 0.36.0 (alternative LLM provider)
- dotenv 16.6.1 (environment config)

---

## 📁 Project Structure

```
d:\Hirely Backend/
├── server.js                    # Express app entry point
├── package.json                 # Dependencies & scripts
├── prisma/
│   ├── schema.prisma            # Database schema (PostgreSQL + Vector DB)
│   └── migrations/              # Database migration history
├── generated/prisma/            # Prisma Client (auto-generated)
├── config/
│   ├── redis.js                 # Redis connection & test utilities
│   └── interviewcache.js        # Interview caching logic
├── routes/
│   ├── interviewRoutes.js       # POST /api/init, /api/submit-answer, /api/next-question
│   ├── evaluateRoutes.js        # POST /api/evaluate (manual evaluation)
│   ├── dashboardRoutes.js       # GET /api/dashboard/* (analytics/reports)
│   └── healthRoutes.js          # GET / (health check)
├── controllers/
│   ├── interviewController.js   # Main interview flow logic
│   ├── evaluateController.js    # Answer evaluation endpoint
│   ├── dashboardController.js   # Dashboard metrics/reports
│   └── healthController.js      # Health check handler
├── services/
│   ├── analyzer.js              # Resume parsing + gap analysis (Gemini)
│   ├── interviewer.js           # Question generation + final report (Gemini)
│   ├── retrieval.js             # Vector search + LLM evaluation
│   ├── stateManager.js          # Redis session management
│   ├── transcriber.js           # Deepgram speech-to-text
│   ├── textNormalizer.js        # Text preprocessing
│   └── tts.js                   # Text-to-speech (Deepgram)
├── supabase/
│   └── config.toml              # Supabase configuration
├── scripts/
│   ├── data_dump.js             # Database export utility
│   ├── ingest.js                # Document ingestion into vector DB
│   ├── check_db.js              # Database connectivity check
│   └── test_full_flow.js        # End-to-end testing
├── samples/                     # Sample data files
├── uploads/                     # Temporary file storage (resumes, audio)
└── database_dump.json           # Database backup/export

```

---

## 🗄️ Database Schema

### Models

#### **Interview**
```prisma
- id: UUID (PK)
- jobDescription: String
- userId: String (Supabase user reference)
- finalScore: Float (aggregated score)
- finalFeedback: JSON (final report)
- createdAt: DateTime
- turns: InterviewTurn[] (relationship)
```

#### **InterviewTurn**
```prisma
- id: Int (PK)
- interviewId: String (FK -> Interview)
- question: String
- answer: String
- score: Int (0-100)
- feedback: String
- improvedAnswer: String (ideal answer)
- topic: String
- difficulty: String
- softSkillScore: Int
- createdAt: DateTime
```

#### **Document** (Vector DB)
```prisma
- id: Int (PK)
- content: String (chunk text)
- embedding: vector(768) (PostgreSQL pgvector)
- metadata: JSON
- source: String
- createdAt: DateTime
```

---

## 🔄 Interview Flow

### 1️⃣ Initialize Interview (`POST /api/init`)
**Input:** Resume (PDF file) + Job Description (text)

**Process:**
1. Parse PDF resume → extract text
2. Call `analyzer.js` → generateInterviewContext()
   - Send resume + JD to Gemini
   - Extract: candidateSummary, gapAnalysis, 4 initial questions
   - Returns: questions array, gap analysis, match score
3. Generate UUID sessionId
4. Store session in Redis with:
   - Job description
   - Initial 4 questions (queue)
   - Gap analysis
   - User ID
   - Current question (Q1)
   - Interview history (empty initially)
5. Generate audio for Q1 (TTS via Deepgram if `ENABLE_TTS=true`)
6. **Response:** 
   ```json
   {
     "sessionId": "uuid",
     "analysis": { gap analysis + summary },
     "firstQuestion": { question, topic, difficulty, reason },
     "audio": "base64 encoded audio"
   }
   ```

---

### 2️⃣ Submit Answer (`POST /api/submit-answer`)
**Input:** sessionId, question, (audio file OR text answer)

**Process:**
1. Transcribe audio → Deepgram if file provided, else use text
2. Call `retrieval.js` → evaluateAnswer()
   - Vector semantic similarity (q vs a)
   - LLM judgment using Gemini
   - Returns: score (0-100), feedback, correctness, betterAnswer
3. Save turn in Redis via stateManager.saveTurn()
   - Append to history array
   - Remove answered question from queue
   - Track: question, answer, score, feedback, improvedAnswer, topic, difficulty
4. Check if interview complete (9 questions max)
   - If YES: Generate final report → save to PostgreSQL → return finalReport
   - If NO: Fetch next question
5. Generate audio for next question
6. **Response:**
   ```json
   {
     "evaluation": { score, feedback, correctness, betterAnswer },
     "nextQuestion": { question, topic, difficulty },
     "audio": "base64 encoded",
     "sessionId": "uuid",
     "completed": false,
     "finalReport": null (or report if complete)
   }
   ```

---

### 3️⃣ Get Next Question (Adaptive)
**Logic in `interviewer.js` → getNextQuestion()**

**Queue Phase:**
- Return pre-generated questions from `session.questionQueue` (Q2-Q4)

**Adaptive Phase** (after Q4):
- Generate new questions dynamically based on:
  - Interview history analysis
  - Weak/strong topic areas
  - Adaptive difficulty adjustment
  - Job description coverage
- Conversational tone with human-like filler words
- May probe deeper or call out short answers

---

### 4️⃣ Generate Final Report
**Triggered after 9 questions or user completion**

**Logic in `interviewer.js` → generateFinalReport()**

**Output Schema:**
```json
{
  "decision": "Strong Hire | Hire | Weak Hire | No Hire",
  "technicalLevel": "Junior | Mid | Senior",
  "summary": "3-sentence performance summary",
  "strengths": ["str1", "str2", ...],
  "weaknesses": ["weak1", "weak2", ...],
  "recommendations": "Improvement feedback"
}
```

**Saved to PostgreSQL:**
- Update Interview model with finalScore + finalFeedback (JSON)

---

## 🔐 Session State Management (Redis)

**Key Pattern:** `interview:{sessionId}`

**Session Object Structure:**
```json
{
  "jobDescription": "...",
  "questionQueue": [
    { "question": "...", "topic": "...", "difficulty": "..." },
    ...
  ],
  "gapAnalysis": { "matchScore": 75, "missingSkills": [...], "feedback": "..." },
  "userId": "supabase-user-id",
  "currentQuestion": { "question": "...", "topic": "...", "difficulty": "..." },
  "history": [
    {
      "question": "...",
      "answer": "...",
      "score": 85,
      "feedback": "...",
      "betterAnswer": "...",
      "topic": "...",
      "difficulty": "...",
      "timestamp": "2026-02-20T10:30:00Z"
    },
    ...
  ],
  "createdAt": "2026-02-20T10:00:00Z"
}
```

**TTL:** 24 hours (86400 seconds)

---

## 📊 Answer Evaluation Pipeline

**File:** `services/retrieval.js`

### Evaluation Stages:

1. **Vector Semantic Similarity**
   - Embed question → vector (768-dim)
   - Embed answer → vector (768-dim)
   - Compute cosine similarity (0-1 range)
   - High similarity = answer directly addresses question

2. **Vector Database Retrieval** (Optional context)
   - Query PostgreSQL vector DB
   - Retrieve top 3 relevant document chunks
   - Combine as "official context"

3. **LLM Judgment**
   - Prompt Gemini 2.5 Flash Lite with:
     - Context (official docs if available)
     - Question
     - Candidate answer
   - Grading rules:
     - Gibberish/unrelated → 0
     - Technically incorrect → 0-20
     - Correct but vague → 40-60
     - Correct + detailed → 80-100
   - Return: score, feedback (2 sentences), correctness, betterAnswer

---

## 🎤 Audio Processing

### Speech-to-Text (Deepgram)
**File:** `services/transcriber.js`
- Input: Audio buffer (wav/mp3)
- Service: Deepgram SDK
- Output: Transcribed text
- Error handling: Fallback to empty string if service fails

### Text-to-Speech (Deepgram)
**File:** `services/tts.js`
- Input: Question text (string)
- Service: Deepgram TTS API
- Output: Audio buffer (base64 encoded for JSON response)
- Configurable: `ENABLE_TTS=true` flag to save credits
- Applied to: Q1 + each subsequent question

---

## 🔌 API Endpoints

### Health Check
- **GET /** → `{ "status": "OK" }`

### Interview Endpoints
- **POST /api/init**
  - Resume upload + job description
  - Returns: sessionId, firstQuestion, audio

- **POST /api/submit-answer**
  - Audio/text answer + sessionId
  - Returns: evaluation, nextQuestion, audio

- **POST /api/next-question**
  - Get next adaptive question
  - Returns: question object

### Evaluation
- **POST /api/evaluate**
  - Manual evaluation: question + answer text
  - Returns: score, feedback, correctness, betterAnswer

### Dashboard
- **GET /api/dashboard/***
  - Metrics, reports, user analytics

---

## 🛡️ Security Features

1. **Rate Limiting**
   - 100 requests per 15 minutes per IP (global)
   - Applied to all routes

2. **CORS**
   - Currently: `origin: "*"` (allow all)
   - Methods: GET, POST only

3. **Input Validation**
   - Zod schema validation on all inputs
   - PDF parsing safety: truncate to 15000 chars
   - File upload via Multer

4. **Environment Variables**
   - All secrets in `.env`:
     - `GOOGLE_API_KEY` (Gemini)
     - `DATABASE_URL` (PostgreSQL)
     - `REDIS_URL` (Redis)
     - `SUPABASE_URL`, `SUPABASE_KEY` (Auth)
     - `DEEPGRAM_API_KEY` (Speech-to-text/TTS)
     - `ENABLE_TTS` (feature flag)

---

## 🚀 Running the Application

### Setup
```bash
npm install
npx prisma migrate dev    # Initialize database + migrations
```

### Development
```bash
npm run dev               # Auto-reload with nodemon
```

### Production
```bash
npm start                 # Run server
```

### Database
```bash
npx prisma studio       # GUI database browser
npx prisma migrate dev   # Apply pending migrations
npx prisma db push       # Sync schema with DB
```

### Scripts
- `scripts/check_db.js` → Verify PostgreSQL connectivity
- `scripts/ingest.js` → Load documents into vector DB
- `scripts/data_dump.js` → Export database to JSON
- `scripts/test_full_flow.js` → E2E testing

---

## 🔗 Integration Points

### External APIs
- **Google Gemini** (Analysis, questions, evaluation, final reports)
- **Deepgram** (Speech-to-text transcription, TTS)
- **Supabase** (Authentication, file storage, SQL database)
- **PostgreSQL** (Persistent data, vector embeddings)
- **Redis** (Session state, caching)

### File Upload Locations
- Resume PDFs → `uploads/`
- Interview recordings → `uploads/`
- Temp processing → `uploads/` (cleaned up after)

---

## 🎓 Key AI Models Used

| Model | Purpose | Provider |
|-------|---------|----------|
| `gemini-2.5-flash` | Gap analysis, question generation, final reports | Google |
| `gemini-2.5-flash-lite` | Answer evaluation (lightweight) | Google |
| `text-embedding-004` | Vector embeddings (768-dim) | Google |
| Deepgram | Speech-to-text, Text-to-speech | Deepgram |
| Xenova Transformers | Local NLP (optional) | Hugging Face |

---

## ⚙️ Configuration & Environment

**.env Variables:**
```env
PORT=4000
GOOGLE_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@host/hirely
REDIS_URL=redis://localhost:6379
SUPABASE_URL=https://xyz.supabase.co
SUPABASE_KEY=eyJ...
DEEPGRAM_API_KEY=...
ENABLE_TTS=true
NODE_ENV=development
```

**Prisma Config:**
- Datasource: PostgreSQL
- Extensions: pgvector
- Output: `generated/prisma/`

---

## 📈 Current Status
- ✅ Core interview flow implemented
- ✅ Resume parsing & gap analysis
- ✅ AI-powered question generation
- ✅ Real-time answer evaluation
- ✅ Redis session management
- ✅ PostgreSQL + Vector DB integration
- ⏳ **TODO:** Setup PostgreSQL and Vector Database (per README)
- ⏳ **TODO:** Production deployment configuration

---

## 🎯 Key Features to Highlight

1. **Adaptive Interviewing**: Questions adjust based on candidate performance
2. **Real-time Grading**: Semantic + LLM-based answer evaluation
3. **Comprehensive Analysis**: Gap analysis + skill assessment + hiring recommendation
4. **Audio Support**: Speech-to-text + text-to-speech for natural interaction
5. **State Persistence**: Redis-backed session management with 24-hour retention
6. **Vector Search**: PostgreSQL pgvector for semantic document retrieval
7. **Multi-format Support**: PDF resumes, audio/text answers
8. **Human-like Interaction**: Conversational interviewer with natural language

---

## 🔍 Critical Dependencies

**Must have for production:**
- PostgreSQL 14+ with pgvector extension
- Redis 5.0+
- Google Cloud API credentials (Gemini)
- Deepgram API account
- Supabase project

---

*Generated: 2026-02-20*
*For detailed implementation specifics, refer to individual service files*
