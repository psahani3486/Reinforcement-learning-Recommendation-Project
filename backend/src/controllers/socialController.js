import Friendship from '../models/Friendship.js';
import User from '../models/User.js';
import Interaction from '../models/Interaction.js';
import Movie from '../models/Movie.js';

export const sendFriendRequest = async (req, res) => {
  try {
    const { recipientId } = req.body;
    if (recipientId === req.user._id.toString()) {
      return res.status(400).json({ message: 'Cannot add yourself' });
    }

    const existing = await Friendship.findOne({
      $or: [
        { requester: req.user._id, recipient: recipientId },
        { requester: recipientId, recipient: req.user._id }
      ]
    });
    if (existing) {
      return res.status(400).json({ message: 'Friendship already exists or pending' });
    }

    const friendship = await Friendship.create({
      requester: req.user._id,
      recipient: recipientId
    });
    res.status(201).json({ success: true, friendship });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const respondFriendRequest = async (req, res) => {
  try {
    const { friendshipId } = req.params;
    const { action } = req.body; // 'accepted' or 'rejected'

    const friendship = await Friendship.findById(friendshipId);
    if (!friendship) return res.status(404).json({ message: 'Not found' });
    if (friendship.recipient.toString() !== req.user._id.toString()) {
      return res.status(403).json({ message: 'Not authorized' });
    }

    friendship.status = action;
    await friendship.save();
    res.json({ success: true, friendship });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getFriends = async (req, res) => {
  try {
    const friendships = await Friendship.find({
      $or: [
        { requester: req.user._id, status: 'accepted' },
        { recipient: req.user._id, status: 'accepted' }
      ]
    }).populate('requester recipient', 'username email');

    const friends = friendships.map(f => {
      const friend = f.requester._id.toString() === req.user._id.toString()
        ? f.recipient : f.requester;
      return { friendshipId: f._id, ...friend.toObject() };
    });

    res.json({ friends });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getPendingRequests = async (req, res) => {
  try {
    const pending = await Friendship.find({
      recipient: req.user._id,
      status: 'pending'
    }).populate('requester', 'username email');

    res.json({ pending });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getFriendActivity = async (req, res) => {
  try {
    const friendships = await Friendship.find({
      $or: [
        { requester: req.user._id, status: 'accepted' },
        { recipient: req.user._id, status: 'accepted' }
      ]
    });

    const friendIds = friendships.map(f =>
      f.requester.toString() === req.user._id.toString() ? f.recipient : f.requester
    );

    const activities = await Interaction.find({ userId: { $in: friendIds } })
      .sort({ timestamp: -1 })
      .limit(20)
      .populate('userId', 'username');

    const movieIds = [...new Set(activities.map(a => a.movieId))];
    const movies = await Movie.find({ movieId: { $in: movieIds } });
    const moviesMap = {};
    movies.forEach(m => { moviesMap[m.movieId] = m; });

    const enriched = activities.map(a => ({
      ...a.toObject(),
      movie: moviesMap[a.movieId]
    }));

    res.json({ activities: enriched });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const searchUsers = async (req, res) => {
  try {
    const { q } = req.query;
    if (!q || q.length < 2) {
      return res.json({ users: [] });
    }

    const users = await User.find({
      _id: { $ne: req.user._id },
      username: { $regex: q, $options: 'i' }
    }).select('username email').limit(10);

    res.json({ users });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const removeFriend = async (req, res) => {
  try {
    const { friendshipId } = req.params;
    await Friendship.findByIdAndDelete(friendshipId);
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};
