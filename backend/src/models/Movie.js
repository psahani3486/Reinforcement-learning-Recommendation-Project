import mongoose from 'mongoose';

const movieSchema = new mongoose.Schema({
  movieId: {
    type: Number,
    required: true,
    unique: true
  },
  title: {
    type: String,
    required: true
  },
  genres: [{
    type: String
  }],
  year: {
    type: Number
  },
  director: String,
  actors: [String],
  plot: String,
  poster: String,
  embedding: {
    type: [Number],
    default: []
  },
  averageRating: {
    type: Number,
    default: 0
  },
  totalRatings: {
    type: Number,
    default: 0
  },
  popularity: {
    type: Number,
    default: 0
  }
}, { timestamps: true });

movieSchema.index({ title: 'text' });
movieSchema.index({ genres: 1 });

export default mongoose.model('Movie', movieSchema);
