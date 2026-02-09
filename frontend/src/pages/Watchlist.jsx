import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bookmark, Trash2, Film } from 'lucide-react';
import toast from 'react-hot-toast';
import Navbar from '../components/Navbar';
import LoadingSpinner from '../components/LoadingSpinner';
import { watchlistService, recommendationService } from '../services/api';

const Watchlist = () => {
  const [watchlist, setWatchlist] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchWatchlist = async () => {
    try {
      const res = await watchlistService.getWatchlist();
      setWatchlist(res.data.watchlist);
    } catch (err) {
      toast.error('Failed to load watchlist');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWatchlist();
  }, []);

  const handleRemove = async (movieId) => {
    try {
      await watchlistService.removeFromWatchlist(movieId);
      setWatchlist(prev => prev.filter(item => item.movieId !== movieId));
      toast.success('Removed from watchlist');
    } catch (err) {
      toast.error('Failed to remove');
    }
  };

  const handleInteraction = async (movie, type) => {
    try {
      await recommendationService.recordInteraction({
        movieId: movie.movieId,
        interactionType: type,
        sessionId: Date.now().toString()
      });
      toast.success(type === 'purchase' ? 'Liked!' : 'Interaction recorded');
    } catch (err) {
      toast.error('Failed');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-dark-900">
        <Navbar />
        <div className="flex justify-center items-center h-[60vh]">
          <LoadingSpinner />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-dark-900">
      <Navbar />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
            <Bookmark className="text-primary-500" />
            My Watchlist
          </h1>
          <p className="text-gray-400">{watchlist.length} movies saved</p>
        </motion.div>

        {watchlist.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-20"
          >
            <Film size={64} className="mx-auto text-gray-600 mb-4" />
            <p className="text-gray-400 text-lg">Your watchlist is empty</p>
            <p className="text-gray-500 text-sm mt-2">Add movies from the dashboard or search page</p>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <AnimatePresence>
              {watchlist.map((item, index) => {
                const movie = item.movie;
                if (!movie) return null;

                return (
                  <motion.div
                    key={item._id}
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    transition={{ delay: index * 0.05 }}
                    className="bg-dark-800 rounded-xl overflow-hidden border border-dark-700 group"
                  >
                    <div className="h-48 overflow-hidden">
                      {movie.poster ? (
                        <img
                          src={movie.poster}
                          alt={movie.title}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                          loading="lazy"
                          onError={(e) => {
                            e.target.style.display = 'none';
                            e.target.nextSibling.style.display = 'flex';
                          }}
                        />
                      ) : null}
                      <div
                        className={`w-full h-full items-center justify-center bg-gradient-to-br from-primary-600/20 to-purple-600/20 ${movie.poster ? 'hidden' : 'flex'}`}
                      >
                        <Film size={48} className="text-primary-500 opacity-50" />
                      </div>
                    </div>

                    <div className="p-4">
                      <h3 className="text-lg font-bold mb-2 line-clamp-1">{movie.title}</h3>

                      {movie.genres && (
                        <div className="flex flex-wrap gap-1.5 mb-3">
                          {movie.genres.slice(0, 3).map((genre, idx) => (
                            <span key={idx} className="bg-primary-600/20 text-primary-400 px-2 py-0.5 rounded-full text-xs">
                              {genre}
                            </span>
                          ))}
                        </div>
                      )}

                      <div className="flex items-center justify-between text-sm text-gray-400 mb-3">
                        <span>⭐ {movie.averageRating?.toFixed(1) || 'N/A'}</span>
                        {movie.year && <span>{movie.year}</span>}
                      </div>

                      <div className="flex gap-2">
                        <motion.button
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          onClick={() => handleInteraction(movie, 'click')}
                          className="flex-1 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg text-sm font-medium transition-colors"
                        >
                          View
                        </motion.button>
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={() => handleRemove(item.movieId)}
                          className="px-3 py-2 bg-red-600/20 hover:bg-red-600 text-red-400 hover:text-white rounded-lg transition-colors"
                        >
                          <Trash2 size={18} />
                        </motion.button>
                      </div>

                      <p className="text-xs text-gray-600 mt-2">
                        Added {new Date(item.addedAt).toLocaleDateString()}
                      </p>
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
};

export default Watchlist;
