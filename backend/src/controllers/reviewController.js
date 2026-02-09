import Review from '../models/Review.js';
import Movie from '../models/Movie.js';

export const createReview = async (req, res) => {
  try {
    const { movieId, rating, text } = req.body;
    if (!rating || rating < 1 || rating > 5) {
      return res.status(400).json({ message: 'Rating must be between 1 and 5' });
    }

    const existing = await Review.findOne({ userId: req.user._id, movieId });
    if (existing) {
      existing.rating = rating;
      existing.text = text || existing.text;
      await existing.save();
      return res.json({ success: true, review: existing, updated: true });
    }

    const review = await Review.create({
      userId: req.user._id,
      movieId,
      rating,
      text: text || ''
    });

    // Update movie average rating
    const allReviews = await Review.find({ movieId });
    const avg = allReviews.reduce((s, r) => s + r.rating, 0) / allReviews.length;
    await Movie.findOneAndUpdate({ movieId }, { averageRating: avg, totalRatings: allReviews.length });

    res.status(201).json({ success: true, review });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getMovieReviews = async (req, res) => {
  try {
    const { movieId } = req.params;
    const { page = 1, limit = 10 } = req.query;
    const skip = (page - 1) * limit;

    const reviews = await Review.find({ movieId: parseInt(movieId) })
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(parseInt(limit))
      .populate('userId', 'username');

    const total = await Review.countDocuments({ movieId: parseInt(movieId) });
    const avgAgg = await Review.aggregate([
      { $match: { movieId: parseInt(movieId) } },
      { $group: { _id: null, avg: { $avg: '$rating' }, count: { $sum: 1 } } }
    ]);

    res.json({
      reviews,
      averageRating: avgAgg[0]?.avg || 0,
      totalReviews: total,
      pagination: { page: parseInt(page), pages: Math.ceil(total / limit) }
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getUserReviews = async (req, res) => {
  try {
    const reviews = await Review.find({ userId: req.user._id })
      .sort({ createdAt: -1 });

    const movieIds = reviews.map(r => r.movieId);
    const movies = await Movie.find({ movieId: { $in: movieIds } });
    const moviesMap = {};
    movies.forEach(m => { moviesMap[m.movieId] = m; });

    const enriched = reviews.map(r => ({
      ...r.toObject(),
      movie: moviesMap[r.movieId]
    }));

    res.json({ reviews: enriched });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const likeReview = async (req, res) => {
  try {
    const { reviewId } = req.params;
    const review = await Review.findById(reviewId);
    if (!review) return res.status(404).json({ message: 'Review not found' });

    const idx = review.likes.indexOf(req.user._id);
    if (idx > -1) {
      review.likes.splice(idx, 1);
    } else {
      review.likes.push(req.user._id);
    }
    await review.save();
    res.json({ success: true, likes: review.likes.length });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const deleteReview = async (req, res) => {
  try {
    const { reviewId } = req.params;
    const review = await Review.findById(reviewId);
    if (!review) return res.status(404).json({ message: 'Review not found' });
    if (review.userId.toString() !== req.user._id.toString()) {
      return res.status(403).json({ message: 'Not authorized' });
    }
    await Review.findByIdAndDelete(reviewId);
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};
