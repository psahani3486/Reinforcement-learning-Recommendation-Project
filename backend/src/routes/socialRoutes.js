import express from 'express';
import { sendFriendRequest, respondFriendRequest, getFriends, getPendingRequests, getFriendActivity, searchUsers, removeFriend } from '../controllers/socialController.js';
import { protect } from '../middleware/auth.js';

const router = express.Router();

router.get('/friends', protect, getFriends);
router.get('/pending', protect, getPendingRequests);
router.get('/activity', protect, getFriendActivity);
router.get('/search', protect, searchUsers);
router.post('/request', protect, sendFriendRequest);
router.put('/respond/:friendshipId', protect, respondFriendRequest);
router.delete('/friend/:friendshipId', protect, removeFriend);

export default router;
