import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, TrendingUp, Award } from 'lucide-react';
import MovieCard from '../components/MovieCard';
import LoadingSpinner from '../components/LoadingSpinner';
import Navbar from '../components/Navbar';
import { recommendationService, sessionService } from '../services/api';
import { useAuthStore } from '../context/authStore';
import toast from 'react-hot-toast';

const Dashboard = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({ totalReward: 0, totalInteractions: 0 });
  const { user } = useAuthStore();
  const sessionRef = useRef(null);
  const [strategy, setStrategy] = useState('rl');

  useEffect(() => {
    const initSession = async () => {
      try {
        const { data } = await sessionService.startSession();
        sessionRef.current = data.session;
        setStrategy(data.session.abGroup || 'rl');
      } catch {}
    };
    initSession();
    fetchRecommendations();
    fetchStats();

    return () => {
      if (sessionRef.current?._id) {
        sessionService.endSession(sessionRef.current._id).catch(() => {});
      }
    };
  }, []);

  const fetchRecommendations = async () => {
    try {
      setLoading(true);
      const sessionId = sessionRef.current?._id || Date.now().toString();
      const { data } = await recommendationService.getRecommendations({
        sessionId,
        limit: 12,
        abGroup: strategy,
      });
      setRecommendations(data.recommendations || []);
      if (data.strategy) setStrategy(data.strategy);
    } catch (error) {
      toast.error('Failed to fetch recommendations');
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    try {
      const { data } = await recommendationService.getStats();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch stats');
    }
  };

  const handleInteraction = async (movie, interactionType) => {
    try {
      const sessionId = sessionRef.current?._id || Date.now().toString();
      const interactionData = {
        movieId: movie.movieId,
        interactionType,
        sessionId,
        position: movie.position,
        rating: interactionType === 'purchase' ? 5 : interactionType === 'skip' ? 1 : 3,
        dwellTime: Math.floor(Math.random() * 60) + 10,
      };

      const { data } = await recommendationService.recordInteraction(interactionData);

      // Record session action
      if (sessionRef.current?._id) {
        sessionService.recordAction(sessionRef.current._id, {
          type: interactionType,
          movieId: movie.movieId,
          data: { position: movie.position, reward: data.reward }
        }).catch(() => {});
      }

      setRecommendations((prev) => prev.filter((m) => m.movieId !== movie.movieId));

      if (interactionType === 'purchase') {
        toast.success(`Added "${movie.title}" to your list! +${data.reward} points`);
      } else if (interactionType === 'skip') {
        toast(`Skipped "${movie.title}"`);
      } else {
        toast(`Clicked "${movie.title}"`);
      }

      fetchStats();

      setRecommendations((prev) => {
        if (prev.length <= 3) {
          fetchRecommendations();
        }
        return prev;
      });
    } catch (error) {
      toast.error('Failed to record interaction');
    }
  };

  return (
    <div className="min-h-screen bg-dark-900">
      <Navbar />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <motion.h1
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="text-4xl font-bold mb-2"
          >
            Welcome back, <span className="gradient-text">{user?.username}</span>!
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="text-gray-400"
          >
            Discover your next favorite movie with AI-powered recommendations
          </motion.p>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="mt-2"
          >
            <span className={`text-xs px-3 py-1 rounded-full ${
              strategy === 'rl' ? 'bg-green-600/20 text-green-400' :
              strategy === 'baseline' ? 'bg-blue-600/20 text-blue-400' :
              'bg-yellow-600/20 text-yellow-400'
            }`}>
              Strategy: {strategy.toUpperCase()}
            </span>
          </motion.div>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4 }}
            className="bg-gradient-to-br from-primary-600 to-primary-700 rounded-xl p-6 shadow-xl"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-primary-100 text-sm font-medium mb-1">Total Reward</p>
                <p className="text-3xl font-bold text-white">{stats.totalReward || 0}</p>
              </div>
              <motion.div
                animate={{ rotate: [0, 10, -10, 0] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <Award size={48} className="text-primary-200" />
              </motion.div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5 }}
            className="bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl p-6 shadow-xl"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-purple-100 text-sm font-medium mb-1">Interactions</p>
                <p className="text-3xl font-bold text-white">{stats.totalInteractions || 0}</p>
              </div>
              <motion.div
                animate={{ scale: [1, 1.1, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <TrendingUp size={48} className="text-purple-200" />
              </motion.div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.6 }}
            className="bg-gradient-to-br from-pink-600 to-pink-700 rounded-xl p-6 shadow-xl"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-pink-100 text-sm font-medium mb-1">Available</p>
                <p className="text-3xl font-bold text-white">{recommendations.length}</p>
              </div>
              <motion.div
                animate={{ rotate: [0, 360] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
              >
                <Sparkles size={48} className="text-pink-200" />
              </motion.div>
            </div>
          </motion.div>
        </div>

        {loading ? (
          <LoadingSpinner size="lg" />
        ) : recommendations.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center py-20"
          >
            <Sparkles size={64} className="mx-auto text-gray-600 mb-4" />
            <h3 className="text-2xl font-bold text-gray-400 mb-2">No more recommendations</h3>
            <p className="text-gray-500">Check back later for more personalized suggestions!</p>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={fetchRecommendations}
              className="mt-6 bg-primary-600 hover:bg-primary-700 text-white px-6 py-3 rounded-lg font-semibold"
            >
              Refresh Recommendations
            </motion.button>
          </motion.div>
        ) : (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
              className="flex items-center justify-between mb-6"
            >
              <h2 className="text-2xl font-bold">Recommended for You</h2>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={fetchRecommendations}
                className="bg-dark-800 hover:bg-dark-700 text-white px-4 py-2 rounded-lg font-medium transition-colors"
              >
                Refresh
              </motion.button>
            </motion.div>

            <AnimatePresence mode="popLayout">
              <motion.div
                layout
                className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
              >
                {recommendations.map((movie, index) => (
                  <MovieCard
                    key={movie.movieId}
                    movie={movie}
                    index={index}
                    onInteraction={handleInteraction}
                  />
                ))}
              </motion.div>
            </AnimatePresence>
          </>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
