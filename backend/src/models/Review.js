import mongoose from 'mongoose';

const reviewSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  movieId: {
    type: Number,
    required: true
  },
  rating: {
    type: Number,
    required: true,
    min: 1,
    max: 5
  },
  text: {
    type: String,
    maxlength: 1000,
    default: ''
  },
  likes: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User'
  }]
}, { timestamps: true });

reviewSchema.index({ userId: 1, movieId: 1 }, { unique: true });
reviewSchema.index({ movieId: 1, createdAt: -1 });

export default mongoose.model('Review', reviewSchema);
