import fs from "fs";
import path from "path";

// Ensure you are running Node v18+ for built-in fetch/FormData
const BASE_URL = "http://localhost:4000/api";

async function simulateInterview() {
  console.log("🚀 Starting Interview Simulation...");

  // 1. CREATE A DUMMY PDF (In memory)
  // We create a simple Blob to mimic a file upload
  const dummyPdf = new Blob(["Dummy PDF Content"], { type: "application/pdf" });

  const pdfBuffer = fs.readFileSync("./samples/resume.pdf");

  const formData = new FormData();
  formData.append(
    "resume",
    new Blob([pdfBuffer], { type: "application/pdf" }),
    "resume.pdf"
  );
  formData.append("jobDescription", "We need a Senior React Engineer.");

  // 2. INIT INTERVIEW
  console.log("\n📨 Sending Resume...");
  const initRes = await fetch(`${BASE_URL}/init-interview`, {
    method: "POST",
    body: formData,
  });
  console.log(initRes.status);

  if (!initRes.ok) throw new Error("Init failed");
  const initData = await initRes.json();

  const sessionId = initData.sessionId;
  let currentQuestion = initData.firstQuestion.question;

  console.log(`✅ Session Started: ${sessionId}`);
  console.log(`\n📝 [Q1] ${currentQuestion}`);

  // 3. THE LOOP (Answer 10 Questions)
  // We loop up to 12 times just to be safe, but it should stop at 10
  for (let i = 1; i <= 12; i++) {
    // Simulate "Thinking" time (optional)
    // await new Promise(r => setTimeout(r, 500));

    // A. Submit Answer
    const answer =
      "I have experience with React and Node.js. I use useEffect for side effects.";
    console.log(`   🗣️  Answering: "${answer.substring(0, 30)}..."`);

    const turnRes = await fetch(`${BASE_URL}/submit-answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId,
        question: currentQuestion,
        answer: answer,
      }),
    });

    const turnData = await turnRes.json();

    // B. Check for Game Over
    if (turnData.isFinished) {
      console.log("\n🏁 INTERVIEW FINISHED!");
      console.log("==================================================");
      console.log("📋 FINAL REPORT:");
      console.log(JSON.stringify(turnData.finalReport, null, 2));
      console.log("==================================================");
      break; // EXIT THE LOOP
    }

    // C. Prepare for next turn
    console.log(`   📊 Score: ${turnData.evaluation.score}/100`);

    if (turnData.nextQuestion) {
      currentQuestion = turnData.nextQuestion.question;
      console.log(`\n📝 [Q${i + 1}] ${currentQuestion}`);
    } else {
      console.log("❌ Error: No next question received?");
      break;
    }
  }
}

simulateInterview().catch(console.error);
