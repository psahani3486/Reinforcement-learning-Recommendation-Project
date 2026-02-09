import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Clock, Star, TrendingUp, Calendar } from 'lucide-react';
import Navbar from '../components/Navbar';
import LoadingSpinner from '../components/LoadingSpinner';
import { recommendationService } from '../services/api';
import toast from 'react-hot-toast';

const History = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [pagination, setPagination] = useState(null);

  useEffect(() => {
    fetchHistory();
  }, [page]);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const { data } = await recommendationService.getHistory({ page, limit: 10 });
      setHistory(data.history);
      setPagination(data.pagination);
    } catch (error) {
      toast.error('Failed to fetch history');
    } finally {
      setLoading(false);
    }
  };

  const getInteractionIcon = (type) => {
    switch (type) {
      case 'purchase':
        return { icon: Star, color: 'text-yellow-500', bg: 'bg-yellow-500/20' };
      case 'click':
        return { icon: TrendingUp, color: 'text-blue-500', bg: 'bg-blue-500/20' };
      case 'skip':
        return { icon: Clock, color: 'text-red-500', bg: 'bg-red-500/20' };
      default:
        return { icon: Calendar, color: 'text-gray-500', bg: 'bg-gray-500/20' };
    }
  };

  const getRewardColor = (reward) => {
    if (reward > 3) return 'text-green-500';
    if (reward > 0) return 'text-blue-500';
    return 'text-red-500';
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
          <h1 className="text-4xl font-bold mb-2 gradient-text">Interaction History</h1>
          <p className="text-gray-400">View your past movie interactions and rewards</p>
        </motion.div>

        {loading ? (
          <LoadingSpinner size="lg" />
        ) : history.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center py-20"
          >
            <Clock size={64} className="mx-auto text-gray-600 mb-4" />
            <h3 className="text-2xl font-bold text-gray-400 mb-2">No history yet</h3>
            <p className="text-gray-500">Start interacting with movies to see your history!</p>
          </motion.div>
        ) : (
          <>
            <div className="space-y-4">
              {history.map((item, index) => {
                const { icon: Icon, color, bg } = getInteractionIcon(item.interactionType);
                const rewardColor = getRewardColor(item.reward);

                return (
                  <motion.div
                    key={item._id}
                    initial={{ opacity: 0, x: -50 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    whileHover={{ scale: 1.02, x: 10 }}
                    className="bg-dark-800 rounded-xl p-6 shadow-lg hover:shadow-2xl transition-all cursor-pointer"
                  >
                    <div className="flex items-center gap-4">
                      <motion.div
                        whileHover={{ rotate: 360 }}
                        transition={{ duration: 0.5 }}
                        className={`${bg} p-3 rounded-full`}
                      >
                        <Icon size={24} className={color} />
                      </motion.div>

                      <div className="flex-1">
                        <h3 className="text-lg font-bold mb-1">
                          {item.movie?.title || `Movie #${item.movieId}`}
                        </h3>
                        <div className="flex items-center gap-4 text-sm text-gray-400">
                          <span className="flex items-center gap-1">
                            <Calendar size={14} />
                            {new Date(item.timestamp).toLocaleDateString()}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock size={14} />
                            {new Date(item.timestamp).toLocaleTimeString()}
                          </span>
                          {item.movie?.genres && (
                            <span className="flex gap-2">
                              {item.movie.genres.slice(0, 2).map((genre) => (
                                <span
                                  key={genre}
                                  className="bg-primary-600/20 text-primary-400 px-2 py-0.5 rounded text-xs"
                                >
                                  {genre}
                                </span>
                              ))}
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="text-right">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-sm text-gray-400 capitalize">
                            {item.interactionType}
                          </span>
                        </div>
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ delay: index * 0.05 + 0.2, type: 'spring' }}
                          className={`text-2xl font-bold ${rewardColor}`}
                        >
                          {item.reward > 0 ? '+' : ''}
                          {item.reward}
                        </motion.div>
                        <span className="text-xs text-gray-500">reward</span>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>

            {pagination && pagination.pages > 1 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center justify-center gap-4 mt-8"
              >
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="bg-dark-800 hover:bg-dark-700 text-white px-6 py-2 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Previous
                </motion.button>

                <span className="text-gray-400">
                  Page {pagination.page} of {pagination.pages}
                </span>

                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setPage((p) => Math.min(pagination.pages, p + 1))}
                  disabled={page === pagination.pages}
                  className="bg-dark-800 hover:bg-dark-700 text-white px-6 py-2 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Next
                </motion.button>
              </motion.div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default History;
