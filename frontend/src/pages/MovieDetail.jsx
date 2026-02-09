import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, Star, Play, Bookmark, BookmarkCheck, Clock, TrendingUp } from 'lucide-react';
import toast from 'react-hot-toast';
import Navbar from '../components/Navbar';
import RatingStars from '../components/RatingStars';
import ReviewCard from '../components/ReviewCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { useAuthStore } from '../context/authStore';
import { recommendationService, reviewService, watchlistService } from '../services/api';

const MovieDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const { user } = useAuthStore();
  const [movie, setMovie] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [avgRating, setAvgRating] = useState(0);
  const [loading, setLoading] = useState(true);
  const [inWatchlist, setInWatchlist] = useState(false);
  const [userRating, setUserRating] = useState(0);
  const [reviewText, setReviewText] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [movieRes, reviewsRes] = await Promise.all([
          recommendationService.getMovieDetails(id),
          reviewService.getMovieReviews(id)
        ]);
        setMovie(movieRes.data.movie);
        setReviews(reviewsRes.data.reviews);
        setAvgRating(reviewsRes.data.averageRating);

        // Check watchlist
        try {
          const wlRes = await watchlistService.checkWatchlist([parseInt(id)]);
          setInWatchlist(wlRes.data.inWatchlist.includes(parseInt(id)));
        } catch {}
      } catch (err) {
        toast.error('Failed to load movie');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [id]);

  const handleWatchlistToggle = async () => {
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

  const handleSubmitReview = async () => {
    if (!userRating) { toast.error('Please select a rating'); return; }
    setSubmitting(true);
    try {
      await reviewService.createReview({
        movieId: movie.movieId,
        rating: userRating,
        text: reviewText
      });
      // Also record as interaction
      await recommendationService.recordInteraction({
        movieId: movie.movieId,
        interactionType: 'rate',
        rating: userRating,
        sessionId: Date.now().toString()
      });
      toast.success('Review submitted!');
      setReviewText('');
      // Refresh reviews
      const res = await reviewService.getMovieReviews(id);
      setReviews(res.data.reviews);
      setAvgRating(res.data.averageRating);
    } catch (err) {
      toast.error('Failed to submit review');
    } finally {
      setSubmitting(false);
    }
  };

  const handleLikeReview = async (reviewId) => {
    try {
      await reviewService.likeReview(reviewId);
      const res = await reviewService.getMovieReviews(id);
      setReviews(res.data.reviews);
    } catch (err) {
      toast.error('Failed');
    }
  };

  const handleDeleteReview = async (reviewId) => {
    try {
      await reviewService.deleteReview(reviewId);
      setReviews(prev => prev.filter(r => r._id !== reviewId));
      toast.success('Review deleted');
    } catch (err) {
      toast.error('Failed to delete');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-dark-900">
        <Navbar />
        <div className="flex justify-center items-center h-[60vh]"><LoadingSpinner /></div>
      </div>
    );
  }

  if (!movie) {
    return (
      <div className="min-h-screen bg-dark-900">
        <Navbar />
        <div className="text-center py-20 text-gray-500">Movie not found</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-dark-900">
      <Navbar />
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Back button */}
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          onClick={() => navigate(-1)}
          className="flex items-center gap-2 text-gray-400 hover:text-white mb-6 transition-colors"
        >
          <ArrowLeft size={20} />
          Back
        </motion.button>

        {/* Movie Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-dark-800 rounded-2xl overflow-hidden border border-dark-700"
        >
          <div className="h-80 relative overflow-hidden">
            {movie.poster ? (
              <img
                src={movie.poster}
                alt={movie.title}
                className="w-full h-full object-cover"
                onError={(e) => {
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'flex';
                }}
              />
            ) : null}
            <div
              className={`w-full h-full items-center justify-center bg-gradient-to-br from-primary-600/30 to-purple-600/30 ${movie.poster ? 'hidden' : 'flex'}`}
            >
              <Play size={80} className="text-primary-500 opacity-30" />
            </div>
            <div className="absolute inset-0 bg-gradient-to-t from-dark-800 via-transparent to-transparent" />
            <div className="absolute top-4 right-4 flex gap-2">
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={handleWatchlistToggle}
                className={`p-3 rounded-xl transition-colors ${
                  inWatchlist
                    ? 'bg-primary-600 text-white'
                    : 'bg-dark-800/80 text-gray-400 hover:text-white'
                }`}
              >
                {inWatchlist ? <BookmarkCheck size={22} /> : <Bookmark size={22} />}
              </motion.button>
            </div>
          </div>

          <div className="p-6">
            <div className="flex items-start justify-between mb-4">
              <h1 className="text-3xl font-bold">{movie.title}</h1>
              <div className="flex items-center gap-2 bg-dark-700 px-3 py-1.5 rounded-lg">
                <Star size={18} className="text-yellow-400 fill-yellow-400" />
                <span className="font-semibold">{avgRating ? avgRating.toFixed(1) : 'N/A'}</span>
              </div>
            </div>

            <div className="flex flex-wrap gap-2 mb-4">
              {movie.genres?.map((genre, idx) => (
                <span key={idx} className="bg-primary-600/20 text-primary-400 px-3 py-1 rounded-full text-sm">
                  {genre}
                </span>
              ))}
              {movie.year && (
                <span className="flex items-center gap-1 bg-dark-700 text-gray-400 px-3 py-1 rounded-full text-sm">
                  <Clock size={14} /> {movie.year}
                </span>
              )}
              {movie.popularity > 0 && (
                <span className="flex items-center gap-1 bg-dark-700 text-gray-400 px-3 py-1 rounded-full text-sm">
                  <TrendingUp size={14} /> {movie.popularity} interactions
                </span>
              )}
            </div>

            {movie.plot && (
              <p className="text-gray-300 leading-relaxed">{movie.plot}</p>
            )}
          </div>
        </motion.div>

        {/* Write Review */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mt-6 bg-dark-800 rounded-xl p-6 border border-dark-700"
        >
          <h3 className="text-lg font-semibold mb-4">Write a Review</h3>
          <div className="mb-4">
            <p className="text-sm text-gray-400 mb-2">Your rating</p>
            <RatingStars rating={userRating} onRate={setUserRating} size={28} />
          </div>
          <textarea
            value={reviewText}
            onChange={e => setReviewText(e.target.value)}
            placeholder="Share your thoughts about this movie..."
            rows={3}
            className="w-full px-4 py-3 bg-dark-700 border border-dark-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 text-white placeholder-gray-500 resize-none"
          />
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleSubmitReview}
            disabled={submitting}
            className="mt-3 px-6 py-2.5 bg-primary-600 hover:bg-primary-700 text-white rounded-xl font-medium transition-colors disabled:opacity-50"
          >
            {submitting ? 'Submitting...' : 'Submit Review'}
          </motion.button>
        </motion.div>

        {/* Reviews */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mt-6"
        >
          <h3 className="text-lg font-semibold mb-4">
            Reviews ({reviews.length})
          </h3>
          <div className="space-y-4">
            {reviews.length === 0 ? (
              <p className="text-center py-8 text-gray-500">No reviews yet. Be the first!</p>
            ) : (
              reviews.map(review => (
                <ReviewCard
                  key={review._id}
                  review={review}
                  onLike={handleLikeReview}
                  onDelete={handleDeleteReview}
                  currentUserId={user?._id}
                />
              ))
            )}
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default MovieDetail;
