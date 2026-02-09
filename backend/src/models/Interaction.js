import mongoose from 'mongoose';

const interactionSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  movieId: {
    type: Number,
    required: true
  },
  interactionType: {
    type: String,
    enum: ['click', 'skip', 'purchase', 'dwell', 'rate'],
    required: true
  },
  rating: {
    type: Number,
    min: 1,
    max: 5
  },
  reward: {
    type: Number,
    required: true
  },
  dwellTime: {
    type: Number,
    default: 0
  },
  sessionId: {
    type: String,
    required: true
  },
  timestamp: {
    type: Date,
    default: Date.now
  },
  metadata: {
    position: Number,
    contextState: mongoose.Schema.Types.Mixed
  }
}, { timestamps: true });

interactionSchema.index({ userId: 1, timestamp: -1 });
interactionSchema.index({ sessionId: 1 });

export default mongoose.model('Interaction', interactionSchema);
