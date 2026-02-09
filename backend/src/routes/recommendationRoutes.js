import express from 'express';
import {
  getRecommendations,
  recordInteraction,
  getUserHistory,
  getMovieDetails,
  getUserStats
} from '../controllers/recommendationController.js';
import { protect } from '../middleware/auth.js';

const router = express.Router();

router.get('/recommendations', protect, getRecommendations);
router.post('/interaction', protect, recordInteraction);
router.get('/history', protect, getUserHistory);
router.get('/stats', protect, getUserStats);
router.get('/movie/:id', protect, getMovieDetails);

export default router;
