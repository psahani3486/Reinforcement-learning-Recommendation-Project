import { motion } from 'framer-motion';
import { Star } from 'lucide-react';
import { useState } from 'react';

const RatingStars = ({ rating = 0, onRate, size = 20, readonly = false }) => {
  const [hoverRating, setHoverRating] = useState(0);

  return (
    <div className="flex items-center gap-1">
      {[1, 2, 3, 4, 5].map((star) => {
        const filled = (hoverRating || rating) >= star;
        return (
          <motion.button
            key={star}
            type="button"
            whileHover={readonly ? {} : { scale: 1.2 }}
            whileTap={readonly ? {} : { scale: 0.9 }}
            onMouseEnter={() => !readonly && setHoverRating(star)}
            onMouseLeave={() => !readonly && setHoverRating(0)}
            onClick={() => !readonly && onRate?.(star)}
            className={`transition-colors ${readonly ? 'cursor-default' : 'cursor-pointer'}`}
            disabled={readonly}
          >
            <Star
              size={size}
              className={filled ? 'text-yellow-400 fill-yellow-400' : 'text-gray-600'}
            />
          </motion.button>
        );
      })}
      {rating > 0 && (
        <span className="ml-2 text-sm text-gray-400">{rating.toFixed(1)}</span>
      )}
    </div>
  );
};

export default RatingStars;
