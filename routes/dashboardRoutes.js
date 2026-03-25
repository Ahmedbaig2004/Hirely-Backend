import express from "express";
import {
  getInterviews,
  getInterviewDetail,
  deleteInterview,
} from "../controllers/dashboardController.js";
const router = express.Router();

router.get("/interviews", getInterviews);
router.get("/interviews/:id", getInterviewDetail);
router.delete("/interviews/:id", deleteInterview);

export default router;
