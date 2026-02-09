# 🎯 Reinforcement Learning Recommendation System

A complete end-to-end movie recommendation system powered by Deep Q-Learning (DQN) with a full-stack web application for real-time recommendations, user interactions, and analytics visualization.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Training Models](#training-models)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This project implements a self-improving recommendation system using Reinforcement Learning (specifically Deep Q-Networks). Unlike traditional collaborative filtering, the RL agent learns to optimize long-term user engagement by:

- **Learning from feedback**: Clicks, purchases, and time spent
- **Balancing exploration vs exploitation**: Discovering new preferences while serving relevant content
- **Adapting to user behavior**: Continuously improving recommendations based on interactions

The system includes:
- **ML Pipeline**: Data preprocessing, baseline models (Matrix Factorization), and DQN training
- **User Simulator**: Realistic environment for RL training based on embedding similarities
- **Full-Stack Web App**: React frontend + Node.js backend for live recommendations
- **MongoDB Integration**: Persistent storage for users, movies, reviews, and interactions

## ✨ Features

### Machine Learning
- ✅ Matrix Factorization baseline (SVD)
- ✅ Custom Gym environment for recommendations
- ✅ Deep Q-Network (DQN) with experience replay
- ✅ Double DQN support
- ✅ User simulator with probabilistic feedback
- ✅ Comprehensive evaluation metrics (Precision@K, NDCG, diversity)

### Web Application
- ✅ User authentication (JWT)
- ✅ Personalized movie recommendations
- ✅ Search and filtering
- ✅ Watchlist management
- ✅ Movie reviews and ratings
- ✅ Social features (friends, shared recommendations)
- ✅ Session tracking and replay
- ✅ Analytics dashboard with visualizations
- ✅ Dark/Light theme toggle
- ✅ Responsive design

## 🏗️ Architecture

```
┌─────────────────┐
│   React App     │  ← User Interface
│   (Frontend)    │
└────────┬────────┘
         │ HTTP/REST
┌────────▼────────┐
│   Express API   │  ← Backend Server
│   (Node.js)     │
└────────┬────────┘
         │ Mongoose
┌────────▼────────┐
│    MongoDB      │  ← Database
└─────────────────┘

┌─────────────────┐
│  Python ML      │  ← RL Training
│  (DQN Agent)    │
└────────┬────────┘
         │
┌────────▼────────┐
│  MovieLens 1M   │  ← Training Data
└─────────────────┘
```

## 🛠️ Tech Stack

### Machine Learning
- **Python 3.8+**
- **NumPy & SciPy**: Numerical computing
- **Gymnasium**: RL environment framework
- **PyYAML**: Configuration management
- **Matplotlib**: Visualization

### Backend
- **Node.js 16+**: Runtime
- **Express.js**: Web framework
- **MongoDB**: NoSQL database
- **Mongoose**: ODM
- **JWT**: Authentication
- **bcrypt**: Password hashing

### Frontend
- **React 18+**: UI framework
- **Vite**: Build tool
- **Tailwind CSS**: Styling
- **Zustand**: State management
- **Axios**: HTTP client
- **React Router**: Navigation

## 📁 Project Structure

```
reinforcement-learning-recommendation-project/
│
├── backend/                    # Node.js API server
│   ├── src/
│   │   ├── server.js          # Entry point
│   │   ├── config/            # Database config
│   │   ├── controllers/       # Route handlers
│   │   ├── middleware/        # Auth middleware
│   │   ├── models/            # Mongoose schemas
│   │   ├── routes/            # API routes
│   │   └── utils/             # Utilities (seeding)
│   └── package.json
│
├── frontend/                   # React web app
│   ├── src/
│   │   ├── App.jsx            # Main component
│   │   ├── components/        # UI components
│   │   ├── pages/             # Route pages
│   │   ├── context/           # State stores
│   │   ├── services/          # API service
│   │   └── utils/             # Helper functions
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
├── src/                        # Python ML package
│   ├── data/                  # Data download & preprocessing
│   ├── models/                # MF & DQN implementations
│   ├── environment/           # Gym env & user simulator
│   ├── training/              # Training scripts
│   └── evaluation/            # Metrics & evaluation
│
├── notebooks/                  # Jupyter notebooks
│   └── 01_quick_start.ipynb   # End-to-end demo
│
├── data/                       # Datasets
│   ├── raw/                   # MovieLens 1M
│   └── processed/             # Train/val/test splits
│
├── results/                    # Model outputs
│   ├── models/                # Saved models
│   └── logs/                  # Training logs
│
├── configs/                    # Configuration files
│   └── config.yaml
│
├── requirements.txt            # Python dependencies
├── setup.py                    # Python package setup
└── README.md                   # This file
```

## 📋 Prerequisites

### For ML Training
- Python 3.8 or higher
- pip package manager

### For Web Application
- Node.js 16 or higher
- npm (comes with Node.js)
- MongoDB (local or cloud instance)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Reinforcement learning Recomandation Project"
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### 3. Set Up Backend

```bash
cd backend
npm install
```

Create `.env` file in `backend/`:
```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/rl-recommendation
JWT_SECRET=your-secret-key-change-in-production
NODE_ENV=development
```

For **MongoDB Atlas** (cloud):
```env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/rl-recommendation?retryWrites=true&w=majority
```

### 4. Set Up Frontend

```bash
cd frontend
npm install
```

Create `.env` file in `frontend/`:
```env
VITE_API_URL=http://localhost:5000/api
VITE_APP_NAME=RL Recommendation System
```

### 5. Download & Prepare Data

```bash
# From project root with venv activated
python -m src.data.download
python -m src.data.preprocess
```

Or use the quick start notebook: `notebooks/01_quick_start.ipynb`

## 📖 Usage

### Running the ML Pipeline

#### Train Baseline Model (Matrix Factorization)

```bash
python -m src.training.train_baseline --config configs/config.yaml
```

#### Train RL Agent (DQN)

```bash
python -m src.training.train_rl \
    --config configs/config.yaml \
    --episodes 1000 \
    --embedding-dim 64
```

#### Evaluate Models

```bash
python -m src.evaluation.evaluate \
    --baseline results/models/baseline.npz \
    --agent results/models/rl/agent.pth
```

### Running the Web Application

#### 1. Start MongoDB (if local)

**Windows**:
```bash
# Start MongoDB service
net start MongoDB
```

**Linux/Mac**:
```bash
sudo systemctl start mongod
```

#### 2. Start Backend Server

```bash
cd backend
npm run dev
```

Backend runs at: `http://localhost:5000`

#### 3. Seed Database (First Time Only)

```bash
cd backend
node src/utils/seedDatabase.js
```

This creates:
- Sample users
- Movie catalog
- Sample reviews and ratings

#### 4. Start Frontend

```bash
cd frontend
npm run dev
```

Frontend runs at: `http://localhost:5173`

### Using the Application

1. **Register/Login**: Create account or use demo credentials
2. **Browse Movies**: Explore movie catalog
3. **Get Recommendations**: View personalized suggestions
4. **Rate & Review**: Provide feedback on movies
5. **Manage Watchlist**: Save movies for later
6. **Social Features**: Connect with friends, share recommendations
7. **Analytics**: View interaction history and statistics

## � Deployment

Deploy the application easily using Vercel (frontend) and Render/Railway (backend):

### Quick Start
1. **Backend**: Deploy to Render or Railway (Node.js + MongoDB)
2. **Frontend**: Deploy to Vercel (React/Vite)
3. **Database**: Use MongoDB Atlas (free tier available)

### Full Instructions

See the detailed deployment guides:
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Step-by-step setup for Vercel + Render/Railway
- **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)** - Quick reference checklist

### Environment Variables

**Backend (Render/Railway)**:
```env
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/rl-recommendation
JWT_SECRET=your-strong-secret-key
NODE_ENV=production
CORS_ORIGIN=https://your-frontend.vercel.app
```

**Frontend (Vercel)**:
```env
VITE_API_URL=https://your-backend.onrender.com
```

### Deployment Platforms
- **Frontend**: [Vercel](https://vercel.com) (free tier)
- **Backend**: [Render](https://render.com) or [Railway](https://railway.app) (free tier)
- **Database**: [MongoDB Atlas](https://mongodb.com/cloud/atlas) (free tier)

## �📡 API Documentation

### Authentication

```bash
POST /api/auth/register
POST /api/auth/login
GET  /api/auth/me
```

### Recommendations

```bash
GET  /api/recommendations       # Get personalized recommendations
GET  /api/recommendations/:id/explain  # Get explanation
POST /api/recommendations/feedback     # Submit feedback
```

### Movies

```bash
GET  /api/search               # Search movies
GET  /api/search/:id           # Get movie details
```

### Reviews

```bash
GET  /api/reviews/movie/:movieId
POST /api/reviews
PUT  /api/reviews/:id
DELETE /api/reviews/:id
```

### Watchlist

```bash
GET  /api/watchlist
POST /api/watchlist/add
DELETE /api/watchlist/remove/:movieId
```

### Social

```bash
GET  /api/social/friends
POST /api/social/friends/add
DELETE /api/social/friends/remove/:friendId
GET  /api/social/recommendations
```

### Analytics

```bash
GET  /api/analytics/overview
GET  /api/analytics/interactions
GET  /api/analytics/preferences
```

### Sessions

```bash
GET  /api/sessions
GET  /api/sessions/:id/replay
```

## 🧪 Training Models

### Configuration

Edit `configs/config.yaml`:

```yaml
data:
  dataset: movielens-1m
  min_user_interactions: 20
  min_item_interactions: 10
  test_size: 0.1
  val_size: 0.1

baseline:
  method: svd
  embedding_dim: 64
  max_iter: 100

rl:
  embedding_dim: 64
  hidden_layers: [256, 128]
  learning_rate: 0.001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 0.995
  buffer_size: 10000
  batch_size: 64
  double_dqn: true

environment:
  max_steps: 20
  num_candidates: 100
  history_length: 10
  purchase_threshold: 0.7
  click_threshold: 0.4
```

### Training Tips

1. **Start with baseline**: Always train Matrix Factorization first to get embeddings
2. **Monitor epsilon**: Track exploration rate during training
3. **Check replay buffer**: Ensure it fills up before heavy training
4. **Use validation**: Monitor performance on validation set
5. **Save checkpoints**: Regularly save agent during training

## 📊 Evaluation

### Metrics Computed

- **Precision@K**: Relevance of top-K recommendations
- **Recall@K**: Coverage of relevant items
- **NDCG@K**: Ranking quality
- **Diversity**: Variety in recommendations
- **Coverage**: Catalog coverage
- **Novelty**: Recommendation of less popular items

### Example Evaluation

```python
from src.evaluation.evaluate import evaluate_model
from src.evaluation.metrics import compute_all_metrics

# Load test data
test_data = load_test_data('data/processed/test.csv')

# Evaluate
results = evaluate_model(agent, test_data, k=10)

print(f"Precision@10: {results['precision']:.4f}")
print(f"NDCG@10: {results['ndcg']:.4f}")
print(f"Diversity: {results['diversity']:.4f}")
```

## 🎓 Key Concepts

### Why Reinforcement Learning?

Traditional recommendation systems optimize for immediate clicks or ratings. RL optimizes for **long-term engagement**:

- **Sequential decision making**: Each recommendation affects future interactions
- **Delayed rewards**: Good recommendations pay off over multiple sessions
- **Exploration**: Discovering new user preferences
- **Adaptability**: Continuously learning from feedback

### User Simulator

The simulator generates realistic user feedback based on:
- **Cosine similarity**: Between user and item embeddings
- **Thresholds**: Purchase (0.7), Click (0.4)
- **Rewards**: Purchase (+5), Click (+2), Skip (-1)
- **Noise**: Probabilistic with temperature parameter

### DQN Architecture

- **State**: User history + current recommendations
- **Action**: Select item from candidate set
- **Reward**: Based on user feedback
- **Q-Network**: Neural network approximating value function
- **Experience Replay**: Breaking correlation in training data
- **Target Network**: Stabilizing training

## 🐛 Troubleshooting

### MongoDB Connection Issues

```bash
# Check if MongoDB is running
# Windows
net start | findstr MongoDB

# Linux/Mac
systemctl status mongod
```

### Port Already in Use

```bash
# Backend (port 5000)
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Frontend (port 5173)
netstat -ano | findstr :5173
taskkill /PID <PID> /F
```

### Python Package Issues

```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Or install package in development mode
pip install -e .
```

### CORS Errors

Ensure backend `server.js` has CORS configured:
```javascript
app.use(cors({
  origin: 'http://localhost:5173',
  credentials: true
}));
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MovieLens Dataset**: GroupLens Research (University of Minnesota)
- **OpenAI Gym**: Environment framework
- **React Community**: UI components and patterns
- **RL Research**: DQN papers and implementations

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ using Reinforcement Learning, React, and Node.js**
