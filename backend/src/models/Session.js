import mongoose from 'mongoose';

const sessionSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  sessionId: {
    type: String,
    required: true,
    unique: true
  },
  actions: [{
    action: { type: String, enum: ['view', 'click', 'skip', 'purchase', 'rate', 'search', 'filter'] },
    movieId: Number,
    reward: Number,
    timestamp: { type: Date, default: Date.now },
    metadata: mongoose.Schema.Types.Mixed
  }],
  totalReward: { type: Number, default: 0 },
  duration: { type: Number, default: 0 },
  startedAt: { type: Date, default: Date.now },
  endedAt: Date,
  abGroup: { type: String, enum: ['rl', 'baseline', 'random'], default: 'rl' }
}, { timestamps: true });

sessionSchema.index({ userId: 1, startedAt: -1 });
sessionSchema.index({ abGroup: 1 });

export default mongoose.model('Session', sessionSchema);
