import { motion } from 'framer-motion';
import { Star, Play, Clock, TrendingUp, Bookmark, BookmarkCheck, Info } from 'lucide-react';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ExplanationPanel from './ExplanationPanel';
import { watchlistService } from '../services/api';
import toast from 'react-hot-toast';

const MovieCard = ({ movie, onInteraction, index }) => {
  const [isHovered, setIsHovered] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [inWatchlist, setInWatchlist] = useState(false);
  const navigate = useNavigate();

  const handleClick = () => {
    if (!isDragging) {
      onInteraction(movie, 'click');
    }
  };

  const handleViewDetail = (e) => {
    e.stopPropagation();
    navigate(`/movie/${movie.movieId}`);
  };

  const handleWatchlistToggle = async (e) => {
    e.stopPropagation();
    try {
      if (inWatchlist) {
        await watchlistService.removeFromWatchlist(movie.movieId);
        setInWatchlist(false);
        toast.success('Removed from watchlist');
      } else {
        await watchlistService.addToWatchlist(movie.movieId);
        setInWatchlist(true);
        toast.success('Added to watchlist');
      }
    } catch (err) {
      toast.error(err.response?.data?.message || 'Failed');
    }
  };

  const handleSkip = (e) => {
    e.stopPropagation();
    onInteraction(movie, 'skip');
  };

  const handlePurchase = (e) => {
    e.stopPropagation();
    onInteraction(movie, 'purchase');
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, x: -100 }}
      transition={{ delay: index * 0.1, duration: 0.5 }}
      drag="x"
      dragConstraints={{ left: 0, right: 0 }}
      dragElastic={0.7}
      onDragStart={() => setIsDragging(true)}
      onDragEnd={(e, info) => {
        setIsDragging(false);
        if (info.offset.x > 100) {
          onInteraction(movie, 'purchase');
        } else if (info.offset.x < -100) {
          onInteraction(movie, 'skip');
        }
      }}
      whileHover={{ scale: 1.03, y: -10 }}
      whileTap={{ scale: 0.98 }}
      className="relative bg-dark-800 rounded-xl overflow-hidden shadow-xl cursor-pointer group"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={handleClick}
    >
      <div className="relative h-64 overflow-hidden">
        <motion.div
          initial={{ scale: 1 }}
          animate={{ scale: isHovered ? 1.1 : 1 }}
          transition={{ duration: 0.3 }}
          className="w-full h-full"
        >
          {movie.poster ? (
            <img
              src={movie.poster}
              alt={movie.title}
              className="w-full h-full object-cover"
              loading="lazy"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'flex';
              }}
            />
          ) : null}
          <div
            className={`w-full h-full items-center justify-center bg-gradient-to-br from-primary-600/20 to-purple-600/20 bg-dark-700 ${movie.poster ? 'hidden' : 'flex'}`}
          >
            <Play size={64} className="text-primary-500 opacity-50" />
          </div>
        </motion.div>

        {movie.score && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="absolute top-4 right-4 bg-primary-600 text-white px-3 py-1 rounded-full text-sm font-bold flex items-center gap-1"
          >
            <TrendingUp size={16} />
            {movie.score.toFixed(0)}
          </motion.div>
        )}

        <motion.button
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={handleWatchlistToggle}
          className="absolute top-4 left-4 p-2 rounded-lg bg-dark-800/70 hover:bg-dark-800 text-gray-400 hover:text-primary-400 transition-colors"
        >
          {inWatchlist ? <BookmarkCheck size={18} className="text-primary-400" /> : <Bookmark size={18} />}
        </motion.button>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: isHovered ? 1 : 0 }}
          className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent flex items-end p-4"
        >
          <div className="w-full flex gap-2">
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={handleSkip}
              className="flex-1 bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded-lg font-semibold transition-colors"
            >
              Skip
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={handleViewDetail}
              className="flex-1 bg-dark-600 hover:bg-dark-500 text-white py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-1"
            >
              <Info size={16} />
              Details
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={handlePurchase}
              className="flex-1 bg-green-600 hover:bg-green-700 text-white py-2 px-4 rounded-lg font-semibold transition-colors"
            >
              Like
            </motion.button>
          </div>
        </motion.div>
      </div>

      <div className="p-4">
        <motion.h3
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-xl font-bold mb-2 line-clamp-1"
        >
          {movie.title}
        </motion.h3>

        {movie.genres && movie.genres.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {movie.genres.slice(0, 3).map((genre, idx) => (
              <motion.span
                key={idx}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.3 + idx * 0.1 }}
                className="bg-primary-600/20 text-primary-400 px-3 py-1 rounded-full text-xs font-medium"
              >
                {genre}
              </motion.span>
            ))}
          </div>
        )}

        <div className="flex items-center justify-between text-sm text-gray-400">
          <div className="flex items-center gap-1">
            <Star size={16} className="text-yellow-500 fill-yellow-500" />
            <span>{movie.averageRating ? movie.averageRating.toFixed(1) : 'N/A'}</span>
          </div>
          {movie.year && (
            <div className="flex items-center gap-1">
              <Clock size={16} />
              <span>{movie.year}</span>
            </div>
          )}
        </div>

        {movie.plot && (
          <motion.p
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: isHovered ? 1 : 0, height: isHovered ? 'auto' : 0 }}
            className="mt-3 text-sm text-gray-400 line-clamp-3"
          >
            {movie.plot}
          </motion.p>
        )}

        {movie.explanation && (
          <ExplanationPanel explanations={movie.explanation} />
        )}
      </div>

      <motion.div
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ delay: 0.5 }}
        className="h-1 bg-gradient-to-r from-primary-600 via-purple-600 to-pink-600"
        style={{ originX: 0 }}
      />
    </motion.div>
  );
};

export default MovieCard;
