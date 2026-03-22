import { prisma } from "../config/db.js";
import {
  checkPythonHealth,
  generateAndVerifyTestCases,
} from "../services/testCaseGenerator.js";
import dotenv from "dotenv";
dotenv.config();

// ─── CLI Arg Parsing ─────────────────────────────────────────────────────────

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = { limit: null, problemId: null, dryRun: false, force: false };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--limit":
        opts.limit = parseInt(args[++i], 10);
        break;
      case "--problem-id":
        opts.problemId = parseInt(args[++i], 10);
        break;
      case "--dry-run":
        opts.dryRun = true;
        break;
      case "--force":
        opts.force = true;
        break;
      default:
        console.warn(`Unknown arg: ${args[i]}`);
    }
  }
  return opts;
}

// ─── Delay ───────────────────────────────────────────────────────────────────

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const opts = parseArgs();
  console.log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("  Test Case Generation & Verification");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log(`  Options: ${JSON.stringify(opts)}`);
  console.log();

  // 1. Health check
  console.log("Checking Python availability...");
  const healthy = await checkPythonHealth();
  if (!healthy) {
    console.error("ABORT: Python not found. Install Python 3 or set PYTHON_CMD env var.");
    process.exit(1);
  }
  console.log();

  // 2. Query problems
  const where = {};
  if (opts.problemId) {
    where.problemId = opts.problemId;
  } else if (!opts.force) {
    where.masterSolution = null;
  }

  const questions = await prisma.codingQuestion.findMany({
    where,
    orderBy: { problemId: "asc" },
    ...(opts.limit ? { take: opts.limit } : {}),
  });

  if (questions.length === 0) {
    console.log("No problems to process. Use --force to re-process existing ones.");
    await prisma.$disconnect();
    return;
  }

  console.log(`Found ${questions.length} problem(s) to process.\n`);

  // 3. Process each problem
  let successCount = 0;
  let failCount = 0;
  let skipCount = 0;
  const failures = [];

  for (let i = 0; i < questions.length; i++) {
    const q = questions[i];
    const label = `[${i + 1}/${questions.length}]`;
    console.log(`${label} ${q.title} (${q.difficulty}) — problemId: ${q.problemId}`);

    try {
      const result = await generateAndVerifyTestCases(q);

      if (result.success) {
        const passedCount = result.verifiedTestCases.length;
        console.log(`  PASS ${passedCount}/30 verified`);

        if (opts.dryRun) {
          console.log(`  [DRY RUN] Would save ${passedCount} test cases + master solution`);
          successCount++;
        } else {
          // Atomic save: delete old test cases + update masterSolution + insert new test cases
          await prisma.$transaction([
            prisma.testCase.deleteMany({
              where: { codingQuestionId: q.id },
            }),
            prisma.codingQuestion.update({
              where: { id: q.id },
              data: { masterSolution: result.masterSolution },
            }),
            prisma.testCase.createMany({
              data: result.verifiedTestCases.map((tc) => ({
                codingQuestionId: q.id,
                input: tc.input,
                expectedOutput: tc.expectedOutput,
                category: tc.category,
                isVerified: true,
              })),
            }),
          ]);
          console.log(`  Saved ${passedCount} test cases to DB`);
          successCount++;
        }
      } else {
        console.log(`  FAIL: ${result.error || "Unknown error"}`);
        failCount++;
        failures.push({
          problemId: q.problemId,
          title: q.title,
          error: result.error,
          failureDetails: result.failures?.slice(0, 5), // Log first 5 failures
        });

        // Log individual failures
        if (result.failures?.length > 0) {
          for (const f of result.failures.slice(0, 3)) {
            console.log(`    - [${f.category}] ${f.statusDescription}: expected="${f.expectedPreview}" actual="${f.actualPreview}"`);
          }
          if (result.failures.length > 3) {
            console.log(`    ... and ${result.failures.length - 3} more failures`);
          }
        }
      }
    } catch (err) {
      console.error(`  ERROR: ${err.message}`);
      failCount++;
      failures.push({
        problemId: q.problemId,
        title: q.title,
        error: err.message,
      });
    }

    // Rate limit delay between problems (skip on last)
    if (i < questions.length - 1) {
      await delay(3000);
    }
  }

  // 4. Summary
  console.log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("  SUMMARY");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log(`  Total:    ${questions.length}`);
  console.log(`  Success:  ${successCount}`);
  console.log(`  Failed:   ${failCount}`);
  if (opts.dryRun) console.log(`  (Dry run — no data was saved)`);
  console.log();

  if (failures.length > 0) {
    console.log("  Failed problems:");
    for (const f of failures) {
      console.log(`    - [${f.problemId}] ${f.title}: ${f.error}`);
    }
    console.log();
  }

  await prisma.$disconnect();
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
