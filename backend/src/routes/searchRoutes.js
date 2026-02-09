import express from 'express';
import { searchMovies, getGenres } from '../controllers/searchController.js';
import { protect } from '../middleware/auth.js';

const router = express.Router();

router.get('/movies', protect, searchMovies);
router.get('/genres', protect, getGenres);

export default router;
