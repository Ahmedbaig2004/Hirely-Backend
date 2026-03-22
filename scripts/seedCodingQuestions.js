import { prisma } from "../config/db.js";
import dotenv from "dotenv";
dotenv.config();

const GITHUB_API_URL =
  "https://api.github.com/repos/neenza/leetcode-problems/contents/problems";
const RAW_BASE_URL =
  "https://raw.githubusercontent.com/neenza/leetcode-problems/master/problems";
const BATCH_SIZE = 10;
const DELAY_MS = 1000;
const MAX_PROBLEMS = 100;

// Priority-ordered DSA pattern list — most specific first
const DSA_PATTERNS = [
  "Trie",
  "Union Find",
  "Segment Tree",
  "Binary Indexed Tree",
  "Topological Sort",
  "Monotonic Stack",
  "Monotonic Queue",
  "Sliding Window",
  "Two Pointers",
  "Binary Search",
  "Divide and Conquer",
  "Dynamic Programming",
  "Backtracking",
  "Greedy",
  "Depth-First Search",
  "Breadth-First Search",
  "Recursion",
  "Heap (Priority Queue)",
  "Graph",
  "Tree",
  "Binary Tree",
  "Binary Search Tree",
  "Linked List",
  "Stack",
  "Queue",
  "Hash Table",
  "Sorting",
  "Bit Manipulation",
  "Math",
  "String",
  "Array",
];

function deriveCategory(topics) {
  if (!topics || topics.length === 0) return "Uncategorized";
  for (const pattern of DSA_PATTERNS) {
    if (topics.includes(pattern)) return pattern;
  }
  return topics[0] ?? "Uncategorized";
}

async function fetchFileList() {
  const headers = { "User-Agent": "hirely-seed-script" };
  if (process.env.GITHUB_TOKEN) {
    headers["Authorization"] = `token ${process.env.GITHUB_TOKEN}`;
  }

  const res = await fetch(`${GITHUB_API_URL}?per_page=100`, { headers });
  if (!res.ok) {
    throw new Error(`GitHub API error: ${res.status} ${res.statusText}`);
  }

  const files = await res.json();
  // Filter only .json files and sort by name to ensure numeric order
  return files
    .filter((f) => f.name.endsWith(".json"))
    .sort((a, b) => a.name.localeCompare(b.name));
}

async function fetchProblemJson(filename) {
  const url = `${RAW_BASE_URL}/${filename}`;
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

async function upsertProblem(problem) {
  const problemId = parseInt(problem.frontend_id ?? problem.problem_id, 10);
  if (isNaN(problemId)) throw new Error("Invalid problemId");

  // Build codeSnippets object: { language_slug: code }
  const codeSnippets = {};
  if (Array.isArray(problem.code_snippets)) {
    for (const snippet of problem.code_snippets) {
      if (snippet.lang_slug && snippet.code) {
        codeSnippets[snippet.lang_slug] = snippet.code;
      }
    }
  } else if (
    problem.code_snippets &&
    typeof problem.code_snippets === "object"
  ) {
    Object.assign(codeSnippets, problem.code_snippets);
  }

  const topics = Array.isArray(problem.topics) ? problem.topics : [];
  const category = deriveCategory(topics);

  await prisma.codingQuestion.upsert({
    where: { problemId },
    update: {
      title: problem.title,
      slug:
        problem.problem_slug ??
        problem.title.toLowerCase().replace(/\s+/g, "-"),
      difficulty: problem.difficulty ?? "Unknown",
      category,
      description: problem.description ?? "",
      topics,
      examples: problem.examples ?? [],
      constraints: Array.isArray(problem.constraints)
        ? problem.constraints
        : [],
      codeSnippets,
      hints: Array.isArray(problem.hints) ? problem.hints : [],
      solution: problem.solution ?? null,
    },
    create: {
      problemId,
      title: problem.title,
      slug:
        problem.problem_slug ??
        problem.title.toLowerCase().replace(/\s+/g, "-"),
      difficulty: problem.difficulty ?? "Unknown",
      category,
      description: problem.description ?? "",
      topics,
      examples: problem.examples ?? [],
      constraints: Array.isArray(problem.constraints)
        ? problem.constraints
        : [],
      codeSnippets,
      hints: Array.isArray(problem.hints) ? problem.hints : [],
      solution: problem.solution ?? null,
    },
  });
}

async function main() {
  console.log("Fetching problem file list from GitHub...");
  const files = await fetchFileList();
  const targetFiles = files.slice(0, MAX_PROBLEMS);
  console.log(
    `Found ${files.length} files. Processing first ${targetFiles.length}...\n`,
  );

  let successCount = 0;
  let failCount = 0;

  for (let i = 0; i < targetFiles.length; i += BATCH_SIZE) {
    const batch = targetFiles.slice(i, i + BATCH_SIZE);

    const fetched = await Promise.allSettled(
      batch.map((f) => fetchProblemJson(f.name)),
    );

    for (const result of fetched) {
      if (result.status === "fulfilled" && result.value) {
        try {
          await upsertProblem(result.value);
          console.log(`  ✔ ${result.value.title}`);
          successCount++;
        } catch (err) {
          console.error(
            `  ✘ ${result.value?.title ?? "unknown"}: ${err.message}`,
          );
          failCount++;
        }
      } else {
        console.error(`  ✘ Failed to fetch file`);
        failCount++;
      }
    }

    console.log(
      `\nBatch ${Math.floor(i / BATCH_SIZE) + 1} done. Progress: ${Math.min(i + BATCH_SIZE, targetFiles.length)}/${targetFiles.length}\n`,
    );

    if (i + BATCH_SIZE < targetFiles.length) {
      await new Promise((r) => setTimeout(r, DELAY_MS));
    }
  }

  console.log(
    `\nSeeding complete. ✔ ${successCount} inserted/updated  ✘ ${failCount} failed`,
  );
  await prisma.$disconnect();
}

main().catch((e) => {
  console.error("Fatal error:", e);
  process.exit(1);
});
