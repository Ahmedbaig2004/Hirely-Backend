import { PrismaClient } from "../generated/prisma/index.js";
import { PrismaPg } from "@prisma/adapter-pg";
import pg from "pg";
import "dotenv/config";

const pool = new pg.Pool({
  connectionString: process.env.DATABASE_DIRECT_URL,
  max: 10,
});
const adapter = new PrismaPg(pool);

export const prisma = new PrismaClient({ adapter });
