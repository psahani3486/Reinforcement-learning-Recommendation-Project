import express from 'express';
import { addToWatchlist, removeFromWatchlist, getWatchlist, checkWatchlist } from '../controllers/watchlistController.js';
import { protect } from '../middleware/auth.js';

const router = express.Router();

router.get('/', protect, getWatchlist);
router.post('/', protect, addToWatchlist);
router.delete('/:movieId', protect, removeFromWatchlist);
router.get('/check', protect, checkWatchlist);

export default router;
