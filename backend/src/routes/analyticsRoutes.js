import express from 'express';
import { getDashboardStats, getCTRTrend, getUserFunnel } from '../controllers/analyticsController.js';
import { protect } from '../middleware/auth.js';

const router = express.Router();

router.get('/dashboard', protect, getDashboardStats);
router.get('/ctr', protect, getCTRTrend);
router.get('/funnel', protect, getUserFunnel);

export default router;
