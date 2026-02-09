import express from 'express';
import { createReview, getMovieReviews, getUserReviews, likeReview, deleteReview } from '../controllers/reviewController.js';
import { protect } from '../middleware/auth.js';

const router = express.Router();

router.post('/', protect, createReview);
router.get('/user', protect, getUserReviews);
router.get('/movie/:movieId', protect, getMovieReviews);
router.put('/:reviewId/like', protect, likeReview);
router.delete('/:reviewId', protect, deleteReview);

export default router;
