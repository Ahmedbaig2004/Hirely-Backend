import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { embedText, embedTexts } from "../config/gemini.js";
import { prisma } from "../config/db.js";
import * as cheerio from "cheerio";
import dotenv from "dotenv";
dotenv.config();

// CONFIG
const PAGE_BATCH_SIZE = 5; // Reduced from 10
const DELAY_MS = 2000; // Increased from 500
const EMBED_BATCH_SIZE = 50; // Max chunks per embedding call

/**
 * 🛡️ FALLBACK URLS
 */
const FALLBACKS = {
  javascript: [
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Closures",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Using_promises",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Event_loop",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise",
    "https://developer.mozilla.org/en-US/docs/Web/API/Document_Object_Model/Introduction",
    "https://javascript.info/async",
    "https://javascript.info/classes",
    "https://javascript.info/prototypes",
  ],
  react: [
    "https://react.dev/learn/thinking-in-react",
    "https://react.dev/reference/react/hooks",
    "https://react.dev/learn/passing-props-to-a-component",
    "https://react.dev/learn/state-a-components-memory",
    "https://react.dev/learn/render-and-commit",
    "https://react.dev/learn/queueing-a-series-of-state-updates",
    "https://react.dev/learn/synchronizing-with-effects",
    "https://react.dev/learn/you-might-not-need-an-effect",
  ],
  typescript: [
    "https://www.typescriptlang.org/docs/handbook/2/everyday-types.html",
    "https://www.typescriptlang.org/docs/handbook/2/narrowing.html",
    "https://www.typescriptlang.org/docs/handbook/2/functions.html",
    "https://www.typescriptlang.org/docs/handbook/2/objects.html",
    "https://www.typescriptlang.org/docs/handbook/2/generics.html",
    "https://www.typescriptlang.org/docs/handbook/2/keyof-types.html",
    "https://www.typescriptlang.org/docs/handbook/utility-types.html",
  ],
  nextjs: [
    "https://nextjs.org/docs/app/building-your-application/routing/defining-routes",
    "https://nextjs.org/docs/app/building-your-application/rendering/server-components",
    "https://nextjs.org/docs/app/building-your-application/rendering/client-components",
    "https://nextjs.org/docs/app/building-your-application/data-fetching/fetching-caching-and-revalidating",
    "https://nextjs.org/docs/app/api-reference/functions/server-actions",
    "https://nextjs.org/docs/app/building-your-application/optimizing/images",
  ],
  sql: [
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-joins/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-inner-join/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-left-join/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-indexes/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-transaction/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-primary-key/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-foreign-key/",
  ],
  design_patterns: [
    "https://www.patterns.dev/vanilla/singleton-pattern/",
    "https://www.patterns.dev/vanilla/observer-pattern/",
    "https://www.patterns.dev/vanilla/factory-pattern/",
    "https://www.patterns.dev/react/container-presentational/",
    "https://www.patterns.dev/react/hooks-pattern/",
    "https://www.patterns.dev/react/provider-pattern/",
  ],
  dsa: [
    "https://www.techinterviewhandbook.org/algorithms/array/",
    "https://www.techinterviewhandbook.org/algorithms/string/",
    "https://www.techinterviewhandbook.org/algorithms/hash-table/",
    "https://www.techinterviewhandbook.org/algorithms/recursion/",
    "https://www.techinterviewhandbook.org/algorithms/sorting-searching/",
    "https://www.techinterviewhandbook.org/algorithms/tree/",
    "https://www.techinterviewhandbook.org/algorithms/graph/",
    "https://www.techinterviewhandbook.org/algorithms/dynamic-programming/",
  ],
};

const targets = [
  {
    xmlUrl: null,
    category: "behavioral",
    mustInclude: [],
    fallback: [
      "https://www.techinterviewhandbook.org/behavioral-interview-questions/",
      "https://www.techinterviewhandbook.org/star-method/",
      "https://www.techinterviewhandbook.org/coding-interview-mistakes/",
    ],
  },
  {
    xmlUrl: "https://react.dev/sitemap-0.xml",
    category: "react",
    mustInclude: ["/learn", "/reference"],
    fallback: FALLBACKS.react,
  },
  {
    xmlUrl: null,
    category: "nextjs",
    mustInclude: [],
    fallback: FALLBACKS.nextjs,
  },
  { xmlUrl: null, category: "sql", mustInclude: [], fallback: FALLBACKS.sql },
  {
    xmlUrl: null,
    category: "typescript",
    mustInclude: [],
    fallback: FALLBACKS.typescript,
  },
  {
    xmlUrl: "https://nodejs.org/sitemap.xml",
    category: "node",
    mustInclude: ["/en/learn", "/en/docs/guides"],
  },
  {
    xmlUrl: "https://docs.docker.com/sitemap.xml",
    category: "devops",
    mustInclude: ["/get-started/", "/manuals/"],
  },
  {
    xmlUrl: null,
    category: "javascript",
    mustInclude: [],
    fallback: FALLBACKS.javascript,
  },
  {
    xmlUrl: null,
    category: "design_patterns",
    mustInclude: [],
    fallback: FALLBACKS.design_patterns,
  },
  { xmlUrl: null, category: "dsa", mustInclude: [], fallback: FALLBACKS.dsa },
];

// ✅ Embed with retry + batching
async function embedWithRetry(texts, retries = 3) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const vectors = await embedTexts(texts);

      // Debug: log what came back
      const emptyCount = vectors.filter((v) => !v || v.length === 0).length;
      if (emptyCount > 0) {
        console.warn(
          `  ⚠️ Got ${emptyCount}/${vectors.length} empty vectors on attempt ${attempt}`,
        );
        throw new Error(`${emptyCount} empty vectors returned`);
      }

      return vectors;
    } catch (err) {
      if (attempt === retries) {
        console.error(
          `  ❌ Embedding failed after ${retries} attempts: ${err.message}`,
        );
        throw err;
      }
      const waitMs = attempt * 3000;
      console.warn(
        `  ⚠️ Embed attempt ${attempt} failed: ${err.message}. Retrying in ${waitMs / 1000}s...`,
      );
      await new Promise((r) => setTimeout(r, waitMs));
    }
  }
}

// ✅ Embed large chunk arrays in safe sub-batches
async function embedInBatches(chunks) {
  const textContents = chunks.map((c) => c.pageContent);
  let allVectors = [];

  for (let i = 0; i < textContents.length; i += EMBED_BATCH_SIZE) {
    const batchTexts = textContents.slice(i, i + EMBED_BATCH_SIZE);
    console.log(
      `  🔢 Embedding sub-batch ${i / EMBED_BATCH_SIZE + 1} (${batchTexts.length} chunks)...`,
    );

    const batchVectors = await embedWithRetry(batchTexts);
    allVectors = [...allVectors, ...batchVectors];

    // Pause between sub-batches to respect RPM limits
    if (i + EMBED_BATCH_SIZE < textContents.length) {
      await new Promise((r) => setTimeout(r, 1500));
    }
  }

  return allVectors;
}

async function getUrlsFromSitemap(xmlUrl, mustIncludeFilters) {
  if (!xmlUrl) return [];
  console.log(`\n🗺️  Fetching XML: ${xmlUrl}`);
  try {
    const response = await fetch(xmlUrl, {
      headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" },
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const xmlText = await response.text();
    const $ = cheerio.load(xmlText, { xmlMode: true });
    let collectedUrls = [];
    const subSitemaps = $("sitemap > loc")
      .map((i, el) => $(el).text())
      .get();
    if (subSitemaps.length > 0) {
      for (const sub of subSitemaps) {
        await new Promise((r) => setTimeout(r, 100));
        const subUrls = await getUrlsFromSitemap(sub, mustIncludeFilters);
        collectedUrls = [...collectedUrls, ...subUrls];
      }
    } else {
      collectedUrls = $("url > loc")
        .map((i, el) => $(el).text())
        .get();
    }
    return collectedUrls.filter((url) =>
      mustIncludeFilters.some((filter) => url.includes(filter)),
    );
  } catch (error) {
    console.warn(
      `  ⚠️  Sitemap failed (${error.message}). Switching to manual fallback.`,
    );
    return [];
  }
}

async function fetchAndCleanPage(url) {
  try {
    const res = await fetch(url, {
      signal: AbortSignal.timeout(10000),
      headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" },
    });
    if (!res.ok) return null;
    const html = await res.text();
    const $ = cheerio.load(html);
    $("nav, footer, aside, script, style, noscript, header, button").remove();
    $(".ad, .on-this-page, .cookie-banner, .search-bar, #docsearch").remove();
    let content =
      $("article").text() ||
      $("main").text() ||
      $(".content").text() ||
      $("body").text();
    content = content.replace(/\s+/g, " ").trim();
    if (content.length < 300) return null;
    const title = $("h1").first().text().trim() || url;
    return { url, title, content };
  } catch (e) {
    return null;
  }
}

async function main() {
  console.log("🚀 Starting Optimized Ingestion...");

  // ✅ Sanity check: confirm embedding API works before doing anything
  console.log("\n🔬 Testing embedding API...");
  try {
    const testVec = await embedText("hello world test");
    if (!testVec || testVec.length === 0)
      throw new Error("Empty vector returned");
    console.log(`✅ Embedding API working! Dimensions: ${testVec.length}\n`);
  } catch (err) {
    console.error(`❌ Embedding API test FAILED: ${err.message}`);
    console.error("Fix your API key or model name before continuing.");
    process.exit(1);
  }

  console.log("🧹 Clearing old data...");
  await prisma.$executeRaw`TRUNCATE TABLE "Document" RESTART IDENTITY CASCADE`;
  console.log("✅ Database Cleaned.\n");

  for (const target of targets) {
    let urls = await getUrlsFromSitemap(target.xmlUrl, target.mustInclude);
    if (urls.length === 0 && target.fallback) urls = target.fallback;
    if (urls.length === 0) continue;

    console.log(
      `\n📂 Processing ${urls.length} pages for [${target.category}]...`,
    );

    for (let i = 0; i < urls.length; i += PAGE_BATCH_SIZE) {
      const batchUrls = urls.slice(i, i + PAGE_BATCH_SIZE);
      const rawPages = await Promise.all(
        batchUrls.map((url) => fetchAndCleanPage(url)),
      );
      const validPages = rawPages.filter((p) => p && p.content.length > 300);

      console.log(
        `  📄 Pages fetched: ${validPages.length}/${batchUrls.length} valid`,
      );

      if (validPages.length > 0) {
        const splitter = new RecursiveCharacterTextSplitter({
          chunkSize: 1000,
          chunkOverlap: 100,
        });
        let batchChunks = [];

        for (const page of validPages) {
          const pageChunks = await splitter.createDocuments(
            [page.content],
            [
              {
                source: page.url,
                title: page.title,
                category: target.category,
              },
            ],
          );
          batchChunks = [...batchChunks, ...pageChunks];
        }

        const sanitizedChunks = batchChunks.filter(
          (c) => c.pageContent && c.pageContent.trim().length > 5,
        );

        console.log(`  🧩 Chunks to embed: ${sanitizedChunks.length}`);

        if (sanitizedChunks.length > 0) {
          try {
            // ✅ Use batched + retried embedding
            const vectors = await embedInBatches(sanitizedChunks);

            let savedCount = 0;
            for (let j = 0; j < sanitizedChunks.length; j++) {
              const chunk = sanitizedChunks[j];
              const vector = vectors[j];

              if (!vector || vector.length === 0) {
                console.warn(
                  `  ⚠️ Skipping empty vector for: ${chunk.metadata.source}`,
                );
                continue;
              }

              await prisma.$executeRaw`
                INSERT INTO "Document" (content, embedding, metadata, source)
                VALUES (
                  ${chunk.pageContent},
                  ${`[${vector.join(",")}]`}::vector,
                  ${JSON.stringify(chunk.metadata)}::jsonb,
                  ${chunk.metadata.source}
                )
              `;
              savedCount++;
            }
            process.stdout.write(`  ✅ Saved ${savedCount} chunks\n`);
          } catch (err) {
            console.error(`\n❌ Batch Embedding Error: ${err.message}`);
          }
        }
      }

      await new Promise((r) => setTimeout(r, DELAY_MS));
    }
    console.log(`\n   ✅ Finished [${target.category}]`);
  }

  console.log("\n🏁 Done! Database populated.");
  await prisma.$disconnect();
}

main().catch((e) => {
  console.error("Fatal Error:", e);
  process.exit(1);
});
