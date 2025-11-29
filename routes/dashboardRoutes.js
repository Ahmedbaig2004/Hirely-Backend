import express from "express";
import {
  getInterviews,
  getInterviewDetail,
} from "../controllers/dashboardController.js";
const router = express.Router();

router.get("/interviews", getInterviews);
router.get("/interviews/:id", getInterviewDetail);

export default router;
