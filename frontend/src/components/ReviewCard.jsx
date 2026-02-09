import { motion } from 'framer-motion';
import { Star, ThumbsUp, Trash2, User } from 'lucide-react';
import RatingStars from './RatingStars';

const ReviewCard = ({ review, onLike, onDelete, currentUserId }) => {
  const isOwner = review.userId?._id === currentUserId || review.userId === currentUserId;
  const hasLiked = review.likes?.includes(currentUserId);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-dark-800 rounded-xl p-4 border border-dark-700"
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-r from-primary-600 to-purple-600 rounded-full flex items-center justify-center">
            <User size={18} className="text-white" />
          </div>
          <div>
            <p className="font-semibold text-sm">
              {review.userId?.username || 'User'}
            </p>
            <p className="text-xs text-gray-500">
              {new Date(review.createdAt).toLocaleDateString()}
            </p>
          </div>
        </div>
        <RatingStars rating={review.rating} readonly size={16} />
      </div>

      {review.text && (
        <p className="mt-3 text-sm text-gray-300 leading-relaxed">{review.text}</p>
      )}

      <div className="mt-3 flex items-center gap-4">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => onLike?.(review._id)}
          className={`flex items-center gap-1.5 text-sm transition-colors ${
            hasLiked ? 'text-primary-400' : 'text-gray-500 hover:text-primary-400'
          }`}
        >
          <ThumbsUp size={16} fill={hasLiked ? 'currentColor' : 'none'} />
          <span>{review.likes?.length || 0}</span>
        </motion.button>

        {isOwner && (
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onDelete?.(review._id)}
            className="text-gray-500 hover:text-red-400 transition-colors"
          >
            <Trash2 size={16} />
          </motion.button>
        )}
      </div>
    </motion.div>
  );
};

export default ReviewCard;
