import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { PrismaClient } from "../generated/prisma/index.js";
import * as cheerio from "cheerio";

const prisma = new PrismaClient();
const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004",
});

// CONFIG
const PAGE_BATCH_SIZE = 10; // Fetch 10 pages at a time
const DELAY_MS = 200; // Pause to be polite to servers

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
    "https://www.typescriptlang.org/docs/handbook/2/generics.html", // Crucial for Senior roles
    "https://www.typescriptlang.org/docs/handbook/2/keyof-types.html",
    "https://www.typescriptlang.org/docs/handbook/utility-types.html", // Pick, Omit, Partial
  ],
  nextjs: [
    "https://nextjs.org/docs/app/building-your-application/routing/defining-routes",
    "https://nextjs.org/docs/app/building-your-application/rendering/server-components", // Server vs Client is #1 question
    "https://nextjs.org/docs/app/building-your-application/rendering/client-components",
    "https://nextjs.org/docs/app/building-your-application/data-fetching/fetching-caching-and-revalidating",
    "https://nextjs.org/docs/app/api-reference/functions/server-actions",
    "https://nextjs.org/docs/app/building-your-application/optimizing/images",
  ],
  sql: [
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-joins/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-inner-join/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-left-join/",
    "https://www.postgresqltutorial.com/postgresql-tutorial/postgresql-indexes/", // Indexing is HUGE in interviews
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
    // Tech Interview Handbook is cleaner than GeeksForGeeks for scraping
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
    xmlUrl: null, // Force Fallback
    category: "nextjs",
    mustInclude: [],
    fallback: FALLBACKS.nextjs,
  },
  {
    xmlUrl: null, // Force Fallback
    category: "sql",
    mustInclude: [],
    fallback: FALLBACKS.sql,
  },
  {
    xmlUrl: null, // Force Fallback
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
    xmlUrl: null, // Force Fallback
    category: "design_patterns",
    mustInclude: [],
    fallback: FALLBACKS.design_patterns,
  },
  {
    xmlUrl: null, // Force Fallback
    category: "dsa",
    mustInclude: [],
    fallback: FALLBACKS.dsa,
  },
];

// 1. Robust Recursive Sitemap Fetcher (FIX #1: REMOVED SLICE LIMIT)
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

    // Check for Sub-Sitemaps
    const subSitemaps = $("sitemap > loc")
      .map((i, el) => $(el).text())
      .get();

    if (subSitemaps.length > 0) {
      console.log(
        `   📂 Found ${subSitemaps.length} sub-sitemaps. Recursively fetching ALL...`
      );

      // FIX #1: We iterate ALL sub-sitemaps now (removed .slice)
      for (const sub of subSitemaps) {
        // Small pause to prevent overwhelming the server during recursion
        await new Promise((r) => setTimeout(r, 100));
        const subUrls = await getUrlsFromSitemap(sub, mustIncludeFilters);
        collectedUrls = [...collectedUrls, ...subUrls];
      }
    } else {
      collectedUrls = $("url > loc")
        .map((i, el) => $(el).text())
        .get();
    }

    const validUrls = collectedUrls.filter((url) =>
      mustIncludeFilters.some((filter) => url.includes(filter))
    );

    return validUrls;
  } catch (error) {
    console.warn(
      `   ⚠️  Sitemap failed (${error.message}). Switching to manual fallback.`
    );
    return [];
  }
}

// 2. Fetch and Clean Page
async function fetchAndCleanPage(url) {
  try {
    const res = await fetch(url, {
      signal: AbortSignal.timeout(10000),
      headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" },
    });
    if (!res.ok) return null;

    const html = await res.text();
    const $ = cheerio.load(html);

    // REMOVE JUNK: Navs, Footers, Sidebars, Scripts
    $("nav, footer, aside, script, style, noscript, header").remove();
    $(".ad, .on-this-page, .cookie-banner, .search-bar, #docsearch").remove();

    // Specifically remove "Ask AI" buttons often found in docs
    $("button").remove();

    // Target Main Content explicitly
    let content =
      $("article").text() ||
      $("main").text() ||
      $(".content").text() ||
      $("body").text();

    // Clean whitespace
    content = content.replace(/\s+/g, " ").trim();

    // Skip if it still looks like navigation garbage
    if (content.includes("Start typing to search") || content.length < 200) {
      return null;
    }

    const title = $("h1").first().text().trim() || url;

    return { url, title, content };
  } catch (e) {
    return null;
  }
}

// 3. Main Loop
async function main() {
  console.log("🚀 Starting Optimized Ingestion...");

  // FIX #3: PREVENT STALENESS (TRUNCATE)
  console.log("🧹 Clearing old data (Full Refresh)...");
  await prisma.$executeRaw`TRUNCATE TABLE "Document" RESTART IDENTITY CASCADE`;
  console.log("✅ Database Cleaned.\n");

  for (const target of targets) {
    let urls = await getUrlsFromSitemap(target.xmlUrl, target.mustInclude);

    if (urls.length === 0 && target.fallback) {
      console.log(`   🛡️  Using Fallback URLs for ${target.category}`);
      urls = target.fallback;
    }

    if (urls.length === 0) continue;

    console.log(
      `   Processing ${urls.length} pages for [${target.category}]...`
    );

    // Process Pages in Batches
    for (let i = 0; i < urls.length; i += PAGE_BATCH_SIZE) {
      const batchUrls = urls.slice(i, i + PAGE_BATCH_SIZE);
      const rawPages = await Promise.all(
        batchUrls.map((url) => fetchAndCleanPage(url))
      );
      const validPages = rawPages.filter((p) => p && p.content.length > 300);

      if (validPages.length > 0) {
        const splitter = new RecursiveCharacterTextSplitter({
          chunkSize: 1000,
          chunkOverlap: 100,
        });

        // Accumulate chunks for the entire batch of pages
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
            ]
          );
          batchChunks = [...batchChunks, ...pageChunks];
        }

        // FIX #2: BATCH EMBEDDING (10x Faster)
        if (batchChunks.length > 0) {
          try {
            // Extract just the text strings for embedding
            const textContents = batchChunks.map((c) => c.pageContent);

            // ONE API call for all chunks in this batch
            const vectors = await embeddings.embedDocuments(textContents);

            // Insert Loop
            let savedCount = 0;
            for (let j = 0; j < batchChunks.length; j++) {
              const chunk = batchChunks[j];
              const vector = vectors[j]; // Match vector to chunk

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
            process.stdout.write(`+${savedCount} `); // Visual progress
          } catch (err) {
            console.error(`\n❌ Batch Embedding Error: ${err.message}`);
          }
        }
      }
      // Delay to respect rate limits
      await new Promise((r) => setTimeout(r, DELAY_MS));
    }
    console.log(`\n   ✅ Finished ${target.category}`);
  }

  console.log("\n🏁 Done! Database populated.");
  await prisma.$disconnect();
}

main().catch((e) => {
  console.error("Fatal Error:", e);
  process.exit(1);
});
