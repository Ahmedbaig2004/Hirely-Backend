// scripts/dump-data.js
import { PrismaClient } from "../generated/prisma/index.js";
import fs from "fs";
import path from "path";

const prisma = new PrismaClient();

async function main() {
  console.log("💾 Fetching raw data from Postgres...");

  // We use raw SQL to select the embedding and cast it to text (::text)
  // so we can see the actual numbers [0.123, -0.98, ...]
  const rows = await prisma.$queryRaw`
    SELECT 
      id, 
      substring(content, 1, 100) as content_preview, 
      metadata, 
      embedding::text as raw_vector 
    FROM "Document" 
    LIMIT 5000;
  `;

  if (rows.length === 0) {
    console.log("❌ Database is empty.");
    return;
  }

  // Define output file path
  const outputPath = path.resolve("database_dump.json");

  // Write to file
  fs.writeFileSync(outputPath, JSON.stringify(rows, null, 2));

  console.log(`\n✅ Data dumped to: ${outputPath}`);
  console.log("   Open this file in VS Code to see your vector data!");

  // Preview one vector dimension
  const sampleVector = JSON.parse(rows[0].raw_vector);
  console.log(`\n🧐 Verification:`);
  console.log(`   ID: ${rows[0].id}`);
  console.log(`   Vector Dimensions: ${sampleVector.length}`);
  console.log(
    `   First 5 numbers: [${sampleVector.slice(0, 5).join(", ")}...]`,
  );

  await prisma.$disconnect();
}

main().catch((e) => console.error(e));
