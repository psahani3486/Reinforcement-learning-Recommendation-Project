import express from 'express';
import { startSession, recordAction, endSession, getSessionReplay, getUserSessions } from '../controllers/sessionController.js';
import { protect } from '../middleware/auth.js';

const router = express.Router();

router.post('/start', protect, startSession);
router.post('/:sessionId/action', protect, recordAction);
router.post('/:sessionId/end', protect, endSession);
router.get('/:sessionId/replay', protect, getSessionReplay);
router.get('/', protect, getUserSessions);

export default router;
