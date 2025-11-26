import express from "express";
import { evaluate } from "../controllers/evaluateController.js";

const router = express.Router();

// Manual evaluation endpoint
router.post("/evaluate", evaluate);

export default router;

