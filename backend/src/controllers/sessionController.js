import Session from '../models/Session.js';
import Movie from '../models/Movie.js';

const AB_GROUPS = ['rl', 'baseline', 'random'];

export const startSession = async (req, res) => {
  try {
    // Assign random A/B group
    const abGroup = AB_GROUPS[Math.floor(Math.random() * AB_GROUPS.length)];

    const session = await Session.create({
      userId: req.user._id,
      abGroup,
      actions: []
    });

    res.status(201).json({ success: true, session });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const recordAction = async (req, res) => {
  try {
    const { sessionId } = req.params;
    const { type, movieId, data } = req.body;

    const session = await Session.findById(sessionId);
    if (!session) return res.status(404).json({ message: 'Session not found' });

    session.actions.push({
      type,
      movieId,
      data: data || {},
      timestamp: new Date()
    });
    await session.save();

    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const endSession = async (req, res) => {
  try {
    const { sessionId } = req.params;
    const session = await Session.findById(sessionId);
    if (!session) return res.status(404).json({ message: 'Session not found' });

    session.endedAt = new Date();
    session.duration = (session.endedAt - session.createdAt) / 1000;
    await session.save();

    res.json({ success: true, session });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getSessionReplay = async (req, res) => {
  try {
    const { sessionId } = req.params;
    const session = await Session.findById(sessionId).populate('userId', 'username');
    if (!session) return res.status(404).json({ message: 'Session not found' });

    const movieIds = [...new Set(session.actions.filter(a => a.movieId).map(a => a.movieId))];
    const movies = await Movie.find({ movieId: { $in: movieIds } });
    const moviesMap = {};
    movies.forEach(m => { moviesMap[m.movieId] = m; });

    const enrichedActions = session.actions.map(a => ({
      ...a.toObject ? a.toObject() : a,
      movie: a.movieId ? moviesMap[a.movieId] : null
    }));

    res.json({
      session: {
        ...session.toObject(),
        actions: enrichedActions
      }
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getUserSessions = async (req, res) => {
  try {
    const { limit = 20, page = 1 } = req.query;
    const skip = (page - 1) * limit;

    const sessions = await Session.find({ userId: req.user._id })
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(parseInt(limit))
      .select('-actions');

    const total = await Session.countDocuments({ userId: req.user._id });

    res.json({
      sessions,
      pagination: { total, page: parseInt(page), pages: Math.ceil(total / limit) }
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};
