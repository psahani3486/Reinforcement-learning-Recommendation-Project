import Interaction from '../models/Interaction.js';
import User from '../models/User.js';
import Movie from '../models/Movie.js';
import Session from '../models/Session.js';

export const getDashboardStats = async (req, res) => {
  try {
    const totalUsers = await User.countDocuments();
    const totalInteractions = await Interaction.countDocuments();
    const totalMovies = await Movie.countDocuments();

    // Interactions per day (last 30 days)
    const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
    const dailyInteractions = await Interaction.aggregate([
      { $match: { timestamp: { $gte: thirtyDaysAgo } } },
      {
        $group: {
          _id: { $dateToString: { format: '%Y-%m-%d', date: '$timestamp' } },
          count: { $sum: 1 },
          avgReward: { $avg: '$reward' }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    // Top movies by interactions
    const topMovies = await Interaction.aggregate([
      { $group: { _id: '$movieId', count: { $sum: 1 }, avgReward: { $avg: '$reward' } } },
      { $sort: { count: -1 } },
      { $limit: 10 }
    ]);
    const topMovieIds = topMovies.map(t => t._id);
    const topMovieDetails = await Movie.find({ movieId: { $in: topMovieIds } });
    const movieMap = {};
    topMovieDetails.forEach(m => { movieMap[m.movieId] = m; });

    const topMoviesEnriched = topMovies.map(t => ({
      ...t,
      movie: movieMap[t._id]
    }));

    // Interaction type distribution
    const interactionDist = await Interaction.aggregate([
      { $group: { _id: '$interactionType', count: { $sum: 1 } } }
    ]);

    // Reward distribution
    const rewardDist = await Interaction.aggregate([
      {
        $bucket: {
          groupBy: '$reward',
          boundaries: [-2, -1, 0, 1, 2, 3, 5, 6],
          default: 'other',
          output: { count: { $sum: 1 } }
        }
      }
    ]);

    // Genre popularity
    const genrePopularity = await Interaction.aggregate([
      {
        $lookup: {
          from: 'movies',
          localField: 'movieId',
          foreignField: 'movieId',
          as: 'movie'
        }
      },
      { $unwind: '$movie' },
      { $unwind: '$movie.genres' },
      { $group: { _id: '$movie.genres', count: { $sum: 1 }, avgReward: { $avg: '$reward' } } },
      { $sort: { count: -1 } },
      { $limit: 10 }
    ]);

    // A/B test results
    const abResults = await Session.aggregate([
      { $match: { abGroup: { $exists: true, $ne: null } } },
      {
        $group: {
          _id: '$abGroup',
          sessions: { $sum: 1 },
          avgDuration: { $avg: '$duration' },
          avgActions: { $avg: { $size: '$actions' } }
        }
      }
    ]);

    res.json({
      overview: { totalUsers, totalInteractions, totalMovies },
      dailyInteractions,
      topMovies: topMoviesEnriched,
      interactionDistribution: interactionDist,
      rewardDistribution: rewardDist,
      genrePopularity,
      abTestResults: abResults
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getCTRTrend = async (req, res) => {
  try {
    const { days = 30 } = req.query;
    const startDate = new Date(Date.now() - parseInt(days) * 24 * 60 * 60 * 1000);

    const ctrData = await Interaction.aggregate([
      { $match: { timestamp: { $gte: startDate } } },
      {
        $group: {
          _id: { $dateToString: { format: '%Y-%m-%d', date: '$timestamp' } },
          total: { $sum: 1 },
          clicks: {
            $sum: { $cond: [{ $in: ['$interactionType', ['click', 'purchase']] }, 1, 0] }
          },
          skips: {
            $sum: { $cond: [{ $eq: ['$interactionType', 'skip'] }, 1, 0] }
          }
        }
      },
      {
        $project: {
          _id: 1,
          total: 1,
          clicks: 1,
          skips: 1,
          ctr: { $cond: [{ $gt: ['$total', 0] }, { $divide: ['$clicks', '$total'] }, 0] }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    res.json({ ctrData });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getUserFunnel = async (req, res) => {
  try {
    const totalUsers = await User.countDocuments();
    const usersWithInteractions = await Interaction.distinct('userId');
    const usersWithClicks = await Interaction.distinct('userId', { interactionType: 'click' });
    const usersWithPurchases = await Interaction.distinct('userId', { interactionType: 'purchase' });
    const usersWithRatings = await Interaction.distinct('userId', { interactionType: 'rate' });

    res.json({
      funnel: [
        { stage: 'Registered', count: totalUsers },
        { stage: 'Active (any interaction)', count: usersWithInteractions.length },
        { stage: 'Clicked', count: usersWithClicks.length },
        { stage: 'Purchased/Liked', count: usersWithPurchases.length },
        { stage: 'Rated', count: usersWithRatings.length }
      ]
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};
