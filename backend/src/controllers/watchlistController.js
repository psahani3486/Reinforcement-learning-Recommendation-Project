import Watchlist from '../models/Watchlist.js';
import Movie from '../models/Movie.js';

export const addToWatchlist = async (req, res) => {
  try {
    const { movieId } = req.body;
    const existing = await Watchlist.findOne({ userId: req.user._id, movieId });
    if (existing) {
      return res.status(400).json({ message: 'Already in watchlist' });
    }
    const item = await Watchlist.create({ userId: req.user._id, movieId });
    res.status(201).json({ success: true, item });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const removeFromWatchlist = async (req, res) => {
  try {
    const { movieId } = req.params;
    await Watchlist.findOneAndDelete({ userId: req.user._id, movieId: parseInt(movieId) });
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getWatchlist = async (req, res) => {
  try {
    const items = await Watchlist.find({ userId: req.user._id }).sort({ addedAt: -1 });
    const movieIds = items.map(i => i.movieId);
    const movies = await Movie.find({ movieId: { $in: movieIds } });
    const moviesMap = {};
    movies.forEach(m => { moviesMap[m.movieId] = m; });

    const watchlist = items.map(item => ({
      ...item.toObject(),
      movie: moviesMap[item.movieId]
    }));

    res.json({ watchlist });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const checkWatchlist = async (req, res) => {
  try {
    const { movieIds } = req.query;
    const ids = movieIds ? movieIds.split(',').map(Number) : [];
    const items = await Watchlist.find({ userId: req.user._id, movieId: { $in: ids } });
    const inWatchlist = items.map(i => i.movieId);
    res.json({ inWatchlist });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};
