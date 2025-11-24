import { PrismaClient } from "../generated/prisma/index.js";

const prisma = new PrismaClient();

async function main() {
  console.log("🔍 Checking Database Health...\n");

  // 1. Total Count
  const total = await prisma.document.count();
  if (total === 0) {
    console.log("❌ Database is EMPTY. Ingestion failed.");
    return;
  }
  console.log(`✅ Total Chunks: ${total}`);

  // 2. Breakdown by Category
  // We use raw SQL because grouping by JSONB fields in Prisma is strict
  const categories = await prisma.$queryRaw`
    SELECT 
      metadata->>'category' as category, 
      COUNT(*) as count 
    FROM "Document" 
    GROUP BY metadata->>'category'
    ORDER BY count DESC
  `;

  console.log("\n📊 Distribution by Category:");
  categories.forEach((row) => {
    // Handle BigInt serialization if necessary
    const count =
      typeof row.count === "bigint" ? row.count.toString() : row.count;
    console.log(`   • ${row.category || "Uncategorized"}: ${count} chunks`);
  });

  // 3. Check for NULL Vectors (Crucial!)
  const nullVectors = await prisma.$queryRaw`
    SELECT COUNT(*) as count FROM "Document" WHERE embedding IS NULL
  `;
  const nullCount = Number(nullVectors[0].count);

  if (nullCount > 0) {
    console.log(`\n⚠️  WARNING: ${nullCount} documents have NULL embeddings!`);
  } else {
    console.log(`\n✅ All ${total} documents have valid vector embeddings.`);
  }

  // 4. Sample Data
  console.log("\n📄 Random Sample (Last 10 entries):");
  const samples = await prisma.document.findMany({
    take: 20,
    orderBy: { id: "desc" },
  });

  samples.forEach((doc) => {
    console.log(
      `   [ID: ${doc.id}] (${doc.metadata?.category}) ${doc.content.substring(
        0,
        70
      )}...`
    );
  });

  await prisma.$disconnect();
}

main().catch((e) => console.error(e));
