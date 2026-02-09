import Movie from '../models/Movie.js';
import Interaction from '../models/Interaction.js';
import User from '../models/User.js';

export const getRecommendations = async (req, res) => {
  try {
    const userId = req.user._id;
    const { sessionId, limit = 10, abGroup } = req.query;

    const user = await User.findById(userId);
    const recentInteractions = await Interaction.find({ userId })
      .sort({ timestamp: -1 })
      .limit(10);

    const interactedMovieIds = recentInteractions.map(i => i.movieId);

    // Collaborative filtering: find similar users
    const similarUserIds = await findSimilarUsers(userId, recentInteractions);
    const collaborativeMovieIds = await getCollaborativeMovies(similarUserIds, interactedMovieIds);

    let recommendations;
    const strategy = abGroup || 'rl';

    if (strategy === 'random') {
      // Random baseline
      recommendations = await Movie.aggregate([
        { $match: { movieId: { $nin: interactedMovieIds } } },
        { $sample: { size: parseInt(limit) } }
      ]);
    } else if (strategy === 'baseline') {
      // Popularity baseline
      recommendations = await Movie.find({ movieId: { $nin: interactedMovieIds } })
        .sort({ popularity: -1 })
        .limit(parseInt(limit));
    } else {
      // RL strategy: genre preference + collaborative + popularity
      const query = { movieId: { $nin: interactedMovieIds } };

      if (user.preferences.genres && user.preferences.genres.length > 0) {
        // Mix genre-preferred and collaborative
        const genreRecs = await Movie.find({
          ...query,
          genres: { $in: user.preferences.genres }
        }).sort({ popularity: -1 }).limit(parseInt(limit));

        const collabRecs = collaborativeMovieIds.length > 0
          ? await Movie.find({ movieId: { $in: collaborativeMovieIds } }).limit(5)
          : [];

        const seen = new Set();
        recommendations = [];
        [...genreRecs, ...collabRecs].forEach(m => {
          const key = m.movieId || m._id.toString();
          if (!seen.has(key)) {
            seen.add(key);
            recommendations.push(m);
          }
        });
        recommendations = recommendations.slice(0, parseInt(limit));
      } else {
        recommendations = await Movie.find(query)
          .sort({ popularity: -1 })
          .limit(parseInt(limit));
      }
    }

    recommendations = recommendations.map((movie, index) => {
      const movieObj = movie.toObject ? movie.toObject() : movie;
      const score = calculateRelevanceScore(movieObj, user, recentInteractions);
      const explanation = generateExplanation(movieObj, user, recentInteractions, collaborativeMovieIds);

      return {
        ...movieObj,
        position: index,
        score,
        explanation,
        strategy
      };
    });

    res.json({
      recommendations,
      sessionId: sessionId || Date.now().toString(),
      userEmbedding: user.embedding,
      strategy
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

async function findSimilarUsers(userId, recentInteractions) {
  if (recentInteractions.length === 0) return [];
  const movieIds = recentInteractions.map(i => i.movieId);

  const similar = await Interaction.aggregate([
    { $match: { movieId: { $in: movieIds }, userId: { $ne: userId } } },
    { $group: { _id: '$userId', commonMovies: { $sum: 1 } } },
    { $sort: { commonMovies: -1 } },
    { $limit: 5 }
  ]);

  return similar.map(s => s._id);
}

async function getCollaborativeMovies(similarUserIds, excludeIds) {
  if (similarUserIds.length === 0) return [];

  const interactions = await Interaction.find({
    userId: { $in: similarUserIds },
    movieId: { $nin: excludeIds },
    interactionType: { $in: ['purchase', 'click'] }
  }).sort({ reward: -1 }).limit(10);

  return [...new Set(interactions.map(i => i.movieId))];
}

function generateExplanation(movie, user, recentInteractions, collaborativeMovieIds) {
  const reasons = [];

  if (user.preferences.genres && movie.genres) {
    const matching = movie.genres.filter(g => user.preferences.genres.includes(g));
    if (matching.length > 0) {
      reasons.push({ type: 'genre', text: `Matches your preferred genres: ${matching.join(', ')}`, weight: 0.4 });
    }
  }

  if (movie.averageRating >= 4) {
    reasons.push({ type: 'rating', text: `Highly rated (${movie.averageRating?.toFixed(1)}/5)`, weight: 0.2 });
  }

  if (movie.popularity > 5) {
    reasons.push({ type: 'popularity', text: 'Popular among users', weight: 0.15 });
  }

  if (collaborativeMovieIds.includes(movie.movieId)) {
    reasons.push({ type: 'collaborative', text: 'Users with similar taste enjoyed this', weight: 0.35 });
  }

  if (reasons.length === 0) {
    reasons.push({ type: 'explore', text: 'Recommended to help you discover new content', weight: 0.1 });
  }

  return reasons;
}

export const recordInteraction = async (req, res) => {
  try {
    const { movieId, interactionType, rating, dwellTime, sessionId, position } = req.body;

    let reward = calculateReward(interactionType, rating, dwellTime);

    const interaction = await Interaction.create({
      userId: req.user._id,
      movieId,
      interactionType,
      rating,
      reward,
      dwellTime,
      sessionId,
      metadata: { position }
    });

    await User.findByIdAndUpdate(req.user._id, {
      $inc: { totalInteractions: 1, totalReward: reward }
    });

    await Movie.findOneAndUpdate(
      { movieId },
      {
        $inc: {
          totalRatings: rating ? 1 : 0,
          popularity: interactionType === 'click' || interactionType === 'purchase' ? 1 : 0
        },
        $set: rating ? { averageRating: await calculateMovieAverageRating(movieId) } : {}
      }
    );

    res.status(201).json({
      success: true,
      interaction,
      reward
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getUserHistory = async (req, res) => {
  try {
    const { limit = 20, page = 1 } = req.query;
    const skip = (page - 1) * limit;

    const interactions = await Interaction.find({ userId: req.user._id })
      .sort({ timestamp: -1 })
      .skip(skip)
      .limit(parseInt(limit));

    const movieIds = [...new Set(interactions.map(i => i.movieId))];
    const movies = await Movie.find({ movieId: { $in: movieIds } });

    const moviesMap = {};
    movies.forEach(movie => {
      moviesMap[movie.movieId] = movie;
    });

    const enrichedHistory = interactions.map(interaction => ({
      ...interaction.toObject(),
      movie: moviesMap[interaction.movieId]
    }));

    const total = await Interaction.countDocuments({ userId: req.user._id });

    res.json({
      history: enrichedHistory,
      pagination: {
        total,
        page: parseInt(page),
        pages: Math.ceil(total / limit)
      }
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getMovieDetails = async (req, res) => {
  try {
    const { id } = req.params;
    const movie = await Movie.findOne({ movieId: parseInt(id) });

    if (!movie) {
      return res.status(404).json({ message: 'Movie not found' });
    }

    const interactionStats = await Interaction.aggregate([
      { $match: { movieId: parseInt(id) } },
      {
        $group: {
          _id: '$interactionType',
          count: { $sum: 1 }
        }
      }
    ]);

    res.json({
      movie,
      stats: interactionStats
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getUserStats = async (req, res) => {
  try {
    const userId = req.user._id;

    const totalInteractions = await Interaction.countDocuments({ userId });

    const interactionsByType = await Interaction.aggregate([
      { $match: { userId } },
      {
        $group: {
          _id: '$interactionType',
          count: { $sum: 1 },
          totalReward: { $sum: '$reward' }
        }
      }
    ]);

    const recentActivity = await Interaction.find({ userId })
      .sort({ timestamp: -1 })
      .limit(7)
      .select('timestamp reward');

    const user = await User.findById(userId).select('-password');

    res.json({
      totalInteractions,
      totalReward: user.totalReward,
      interactionsByType,
      recentActivity,
      preferences: user.preferences
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

function calculateReward(interactionType, rating, dwellTime) {
  let reward = 0;

  switch (interactionType) {
    case 'purchase':
      reward = 5;
      break;
    case 'dwell':
      reward = dwellTime > 30 ? 2 : 1;
      break;
    case 'click':
      reward = 1;
      break;
    case 'rate':
      reward = rating >= 4 ? 5 : rating === 3 ? 2 : -1;
      break;
    case 'skip':
      reward = -1;
      break;
    default:
      reward = 0;
  }

  return reward;
}

function calculateRelevanceScore(movie, user, recentInteractions) {
  let score = movie.popularity || 0;

  if (user.preferences.genres) {
    const genreMatch = movie.genres.filter(g =>
      user.preferences.genres.includes(g)
    ).length;
    score += genreMatch * 10;
  }

  score += movie.averageRating * 5;

  return score;
}

async function calculateMovieAverageRating(movieId) {
  const interactions = await Interaction.find({
    movieId,
    rating: { $exists: true, $ne: null }
  });

  if (interactions.length === 0) return 0;

  const sum = interactions.reduce((acc, curr) => acc + curr.rating, 0);
  return sum / interactions.length;
}
