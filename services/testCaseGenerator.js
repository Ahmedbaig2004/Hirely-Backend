import { generateStructured } from "../config/gemini.js";
import { z } from "zod";
import { execFile } from "node:child_process";
import { writeFile, unlink, mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import dotenv from "dotenv";
dotenv.config();

// ─── Constants ───────────────────────────────────────────────────────────────

const PYTHON_CMD = process.env.PYTHON_CMD || "python";
const TIMEOUT_MS = 10000; // 10s timeout per test case execution
const BATCH_SIZE = 5;
const BATCH_DELAY_MS = 200;
const PASS_THRESHOLD = 28;

// ─── Zod Schema ──────────────────────────────────────────────────────────────

const TestCaseSchema = z.object({
  input: z.string().describe("Exact stdin string for this test case"),
  expectedOutput: z.string().describe("Exact stdout string the solution should produce"),
  category: z.enum(["basic", "edge", "stress"]).describe("Test case category"),
  rationale: z.string().describe("Why this test case is important"),
});

const GenerationSchema = z.object({
  ioFormat: z.object({
    inputDescription: z.string().describe("How stdin is formatted"),
    outputDescription: z.string().describe("How stdout is formatted"),
  }),
  masterSolution: z.string().describe("Complete Python 3 solution reading from stdin, printing to stdout"),
  testCases: z.array(TestCaseSchema).length(30).describe("Exactly 30 test cases: 10 basic, 10 edge, 10 stress"),
});

// ─── Helpers ─────────────────────────────────────────────────────────────────

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function formatExamples(examples) {
  if (!examples || !Array.isArray(examples)) return "None provided.";
  return examples
    .map((ex) => {
      if (typeof ex === "string") return ex;
      return ex.example_text || JSON.stringify(ex);
    })
    .join("\n\n");
}

function sanitizeSolution(code) {
  if (!code) return code;
  let cleaned = code.trim();
  if (cleaned.startsWith("```")) {
    cleaned = cleaned.replace(/^```(?:python3?|py)?\s*\n?/, "");
    cleaned = cleaned.replace(/\n?```\s*$/, "");
  }
  return cleaned.trim();
}

function validateGeneration(result) {
  const errors = [];

  if (!result.masterSolution) {
    errors.push("masterSolution is empty");
  } else {
    const sol = result.masterSolution;
    if (!sol.includes("input(") && !sol.includes("sys.stdin") && !sol.includes("stdin")) {
      errors.push("masterSolution does not read from stdin");
    }
    if (!sol.includes("print(") && !sol.includes("sys.stdout")) {
      errors.push("masterSolution does not print to stdout");
    }
  }

  if (!result.testCases || result.testCases.length !== 30) {
    errors.push(`Expected 30 test cases, got ${result.testCases?.length ?? 0}`);
  } else {
    const counts = { basic: 0, edge: 0, stress: 0 };
    for (const tc of result.testCases) {
      counts[tc.category]++;
      if (!tc.input && tc.input !== "") errors.push(`Test case has undefined input`);
      if (!tc.expectedOutput && tc.expectedOutput !== "") errors.push(`Test case has undefined expectedOutput`);
    }
    if (counts.basic !== 10) errors.push(`Expected 10 basic, got ${counts.basic}`);
    if (counts.edge !== 10) errors.push(`Expected 10 edge, got ${counts.edge}`);
    if (counts.stress !== 10) errors.push(`Expected 10 stress, got ${counts.stress}`);
  }

  return { valid: errors.length === 0, errors };
}

// ─── Prompt Builder ──────────────────────────────────────────────────────────

function buildPrompt(question, retryContext = null) {
  const pythonSnippet =
    question.codeSnippets?.python3 ||
    question.codeSnippets?.python ||
    null;

  let prompt = `You are an expert competitive programming problem setter. Create a complete Python 3 solution and exactly 30 test cases for this problem.

## PROBLEM
Title: ${question.title}
Difficulty: ${question.difficulty}
Category: ${question.category}

### Description
${question.description}

### Examples
${formatExamples(question.examples)}

### Constraints
${(question.constraints || []).join("\n")}
${pythonSnippet ? `\n### Function Signature (Reference)\n${pythonSnippet}` : ""}

## YOUR TASK

1. **Design a stdin/stdout I/O format** for this problem following these STRICT conventions:
   - For a single integer parameter: one line with the integer
   - For a single string parameter: one line with the string
   - For List[int] / List[float]: first line = n (length), second line = n space-separated values
   - For List[str]: first line = n, next n lines = one string each
   - For List[List[int]] (2D array / matrix): first line = m (rows), then for each row: one line with space-separated values. All rows have the same number of columns.
   - For multiple parameters: one parameter per block in the order they appear in the function signature, each following the rules above
   - For linked lists: space-separated node values on one line
   - For binary trees: space-separated level-order traversal with "null" for empty nodes
   - Output: print the return value. Arrays → space-separated on one line. Booleans → "true" or "false" (lowercase). Strings → the string itself. Numbers → the number itself. List of lists → one sublist per line, space-separated.

2. **Write a master Python 3 solution** that:
   - Reads input from stdin using the format you designed
   - Implements the OPTIMAL algorithm (best known time complexity for this problem)
   - Prints the result to stdout with NO trailing whitespace, NO extra blank lines
   - Uses \`import sys; input = sys.stdin.readline\` for fast I/O
   - Handles ALL edge cases without crashing (empty inputs, single elements, etc.)
   - Is DETERMINISTIC — if the problem has multiple valid answers, sort or pick the lexicographically smallest

3. **Generate exactly 30 test cases** split as:
   - **10 "basic"**: Small inputs testing normal behavior, including all provided examples from the problem
   - **10 "edge"**: Boundary conditions — empty inputs, single elements, all-same values, maximum/minimum constraint values, negative numbers, zero
   - **10 "stress"**: Large inputs near constraint upper bounds. For stress cases, use arrays up to 50,000 elements (NOT 100,000 — keep the JSON response manageable). Design inputs so O(n²) or worse would exceed 2 seconds, but optimal O(n log n) or O(n) finishes quickly.

## CRITICAL RULES
- Each test case's \`input\` must be the EXACT string to feed to stdin (lines separated by \\n)
- Each test case's \`expectedOutput\` must be the EXACT string the solution prints (NO trailing newline)
- Your master solution MUST produce the exact expectedOutput for EVERY test case — double-check each one
- For stress cases, generate ACTUAL full data — do NOT use placeholders like "..."
- Input values MUST respect the constraints
- Ensure exactly 10 basic, 10 edge, and 10 stress test cases`;

  if (retryContext) {
    prompt += `\n\n## RETRY — PREVIOUS ATTEMPT FAILED
The following test cases from your previous attempt failed verification:
${retryContext}

Fix the master solution AND regenerate test cases. Common issues:
- Output format mismatch (trailing spaces, wrong true/false capitalization)
- Algorithm bug on edge cases
- Input format the solution cannot parse
- Off-by-one errors in expectedOutput`;
  }

  return prompt;
}

// ─── Local Python Execution ──────────────────────────────────────────────────

/**
 * Check that Python is available locally.
 */
export async function checkPythonHealth() {
  return new Promise((resolve) => {
    execFile(PYTHON_CMD, ["--version"], { timeout: 5000 }, (err, stdout, stderr) => {
      if (err) {
        console.error(`Python not found (command: "${PYTHON_CMD}"): ${err.message}`);
        console.error(`Set PYTHON_CMD env var if python is at a different path.`);
        resolve(false);
        return;
      }
      const version = (stdout || stderr).trim();
      console.log(`Found: ${version}`);
      resolve(true);
    });
  });
}

/**
 * Run a Python solution with given stdin, return { passed, stdout, stderr, time }.
 * Compares stdout against expectedOutput (trimmed).
 */
function runPython(solutionPath, stdin, expectedOutput) {
  return new Promise((resolve) => {
    const start = performance.now();
    const child = execFile(
      PYTHON_CMD,
      [solutionPath],
      { timeout: TIMEOUT_MS, maxBuffer: 10 * 1024 * 1024 },
      (err, stdout, stderr) => {
        const elapsed = ((performance.now() - start) / 1000).toFixed(3);

        if (err) {
          if (err.killed) {
            resolve({
              passed: false,
              statusDescription: `Time Limit Exceeded (>${TIMEOUT_MS / 1000}s)`,
              stdout: null,
              stderr: null,
              time: elapsed,
            });
          } else {
            resolve({
              passed: false,
              statusDescription: `Runtime Error: ${err.message.split("\n")[0]}`,
              stdout: stdout?.trim() || null,
              stderr: stderr?.trim()?.slice(0, 300) || null,
              time: elapsed,
            });
          }
          return;
        }

        const actual = stdout.trim();
        const expected = expectedOutput.trim();
        const passed = actual === expected;

        resolve({
          passed,
          statusDescription: passed ? "Accepted" : "Wrong Answer",
          stdout: actual,
          stderr: stderr?.trim() || null,
          time: elapsed,
        });
      }
    );

    if (child.stdin) {
      child.stdin.write(stdin);
      child.stdin.end();
    }
  });
}

/**
 * Verify all test cases by running the master solution with local Python.
 */
async function verifyAllTestCases(masterSolution, testCases) {
  const tmpDir = await mkdtemp(join(tmpdir(), "hirely-tc-"));
  const solutionPath = join(tmpDir, "solution.py");
  await writeFile(solutionPath, masterSolution, "utf-8");

  const results = [];

  try {
    for (let i = 0; i < testCases.length; i += BATCH_SIZE) {
      const batch = testCases.slice(i, i + BATCH_SIZE);
      const batchResults = await Promise.all(
        batch.map((tc) => runPython(solutionPath, tc.input, tc.expectedOutput))
      );

      for (let j = 0; j < batchResults.length; j++) {
        results.push({
          ...batchResults[j],
          testCase: batch[j],
          index: i + j,
        });
      }

      if (i + BATCH_SIZE < testCases.length) {
        await delay(BATCH_DELAY_MS);
      }
    }
  } finally {
    try { await unlink(solutionPath); } catch { /* ignore */ }
  }

  return results;
}

// ─── Gemini Call ─────────────────────────────────────────────────────────────

async function callGemini(question, retryContext = null) {
  const prompt = buildPrompt(question, retryContext);

  const backoffs = [0, 5000, 15000, 45000];
  for (let attempt = 0; attempt < backoffs.length; attempt++) {
    if (backoffs[attempt] > 0) {
      console.log(`    Gemini rate limit — waiting ${backoffs[attempt] / 1000}s...`);
      await delay(backoffs[attempt]);
    }
    try {
      return await generateStructured(prompt, GenerationSchema, { temperature: 0.2 });
    } catch (err) {
      const isRateLimit = err.message?.includes("429") || err.message?.includes("RESOURCE_EXHAUSTED");
      if (isRateLimit && attempt < backoffs.length - 1) continue;
      throw err;
    }
  }
}

// ─── Main Export ─────────────────────────────────────────────────────────────

export async function generateAndVerifyTestCases(question) {
  let generation;
  try {
    generation = await callGemini(question);
  } catch (err) {
    return {
      success: false,
      error: `Gemini call failed: ${err.message}`,
      masterSolution: null,
      verifiedTestCases: [],
      failures: [],
    };
  }

  generation.masterSolution = sanitizeSolution(generation.masterSolution);

  const validation = validateGeneration(generation);
  if (!validation.valid) {
    console.log(`    Validation errors: ${validation.errors.join(", ")}`);
    try {
      const retryCtx = `Structural validation failed:\n${validation.errors.map((e) => `- ${e}`).join("\n")}`;
      generation = await callGemini(question, retryCtx);
      generation.masterSolution = sanitizeSolution(generation.masterSolution);
      const revalidation = validateGeneration(generation);
      if (!revalidation.valid) {
        return {
          success: false,
          error: `Validation failed after retry: ${revalidation.errors.join(", ")}`,
          masterSolution: null,
          verifiedTestCases: [],
          failures: [],
        };
      }
    } catch (err) {
      return {
        success: false,
        error: `Gemini retry failed: ${err.message}`,
        masterSolution: null,
        verifiedTestCases: [],
        failures: [],
      };
    }
  }

  console.log(`    Verifying 30 test cases with local Python...`);
  const results = await verifyAllTestCases(
    generation.masterSolution,
    generation.testCases
  );

  const passed = results.filter((r) => r.passed);
  const failed = results.filter((r) => !r.passed);
  console.log(`    Results: ${passed.length}/30 passed`);

  if (passed.length >= PASS_THRESHOLD) {
    return {
      success: true,
      masterSolution: generation.masterSolution,
      verifiedTestCases: passed.map((r) => ({
        input: r.testCase.input,
        expectedOutput: r.testCase.expectedOutput,
        category: r.testCase.category,
      })),
      failures: failed.map(formatFailure),
    };
  }

  console.log(`    Below threshold (${passed.length}/${PASS_THRESHOLD}). Retrying...`);
  const retryContext = failed
    .map((f) => {
      const inputPreview = f.testCase.input.length > 200
        ? f.testCase.input.slice(0, 200) + "..."
        : f.testCase.input;
      return `- [${f.testCase.category}] ${f.statusDescription}
  Input: ${inputPreview}
  Expected: ${f.testCase.expectedOutput?.slice(0, 100) || "(empty)"}
  Actual: ${f.stdout?.slice(0, 100) || "(no output)"}${f.stderr ? `\n  Stderr: ${f.stderr.slice(0, 100)}` : ""}`;
    })
    .join("\n");

  let retryGeneration;
  try {
    retryGeneration = await callGemini(question, retryContext);
  } catch (err) {
    return {
      success: false,
      error: `Gemini retry failed: ${err.message}`,
      masterSolution: generation.masterSolution,
      verifiedTestCases: passed.map((r) => ({
        input: r.testCase.input,
        expectedOutput: r.testCase.expectedOutput,
        category: r.testCase.category,
      })),
      failures: failed.map(formatFailure),
    };
  }

  retryGeneration.masterSolution = sanitizeSolution(retryGeneration.masterSolution);

  const retryValidation = validateGeneration(retryGeneration);
  if (!retryValidation.valid) {
    return {
      success: false,
      error: `Retry validation failed: ${retryValidation.errors.join(", ")}`,
      masterSolution: generation.masterSolution,
      verifiedTestCases: passed.map((r) => ({
        input: r.testCase.input,
        expectedOutput: r.testCase.expectedOutput,
        category: r.testCase.category,
      })),
      failures: failed.map(formatFailure),
    };
  }

  console.log(`    Re-verifying 30 test cases (retry)...`);
  const retryResults = await verifyAllTestCases(
    retryGeneration.masterSolution,
    retryGeneration.testCases
  );

  const retryPassed = retryResults.filter((r) => r.passed);
  const retryFailed = retryResults.filter((r) => !r.passed);
  console.log(`    Retry results: ${retryPassed.length}/30 passed`);

  if (retryPassed.length >= PASS_THRESHOLD) {
    return {
      success: true,
      masterSolution: retryGeneration.masterSolution,
      verifiedTestCases: retryPassed.map((r) => ({
        input: r.testCase.input,
        expectedOutput: r.testCase.expectedOutput,
        category: r.testCase.category,
      })),
      failures: retryFailed.map(formatFailure),
    };
  }

  return {
    success: false,
    error: `Failed after retry: ${retryPassed.length}/30 passed (need ${PASS_THRESHOLD})`,
    masterSolution: retryGeneration.masterSolution,
    verifiedTestCases: retryPassed.map((r) => ({
      input: r.testCase.input,
      expectedOutput: r.testCase.expectedOutput,
      category: r.testCase.category,
    })),
    failures: retryFailed.map(formatFailure),
  };
}

function formatFailure(f) {
  return {
    index: f.index,
    category: f.testCase.category,
    statusDescription: f.statusDescription,
    inputPreview: f.testCase.input?.slice(0, 150) || "",
    expectedPreview: f.testCase.expectedOutput?.slice(0, 100) || "",
    actualPreview: f.stdout?.slice(0, 100) || "",
    stderr: f.stderr?.slice(0, 200) || "",
  };
}
