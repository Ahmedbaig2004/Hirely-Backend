import "dotenv/config";
import { defineConfig, env } from "prisma/config";

export default defineConfig({
  schema: "prisma/schema.prisma",
  datasource: {
    // This is for your application logic (Transaction Pooler)
    url: env("DATABASE_DIRECT_URL"),
    
  },
});