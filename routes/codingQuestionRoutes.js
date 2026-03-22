import express from "express";
import {
  getCodingQuestions,
  getCodingQuestionById,
} from "../controllers/codingQuestionController.js";

const router = express.Router();

router.get("/coding-questions", getCodingQuestions);
router.get("/coding-questions/:id", getCodingQuestionById);

export default router;
