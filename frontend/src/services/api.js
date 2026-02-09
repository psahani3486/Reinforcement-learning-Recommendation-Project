import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const authService = {
  register: (userData) => api.post('/auth/register', userData),
  login: (credentials) => api.post('/auth/login', credentials),
  getProfile: () => api.get('/auth/profile'),
  updatePreferences: (preferences) => api.put('/auth/preferences', preferences),
};

export const recommendationService = {
  getRecommendations: (params) => api.get('/recommend/recommendations', { params }),
  recordInteraction: (data) => api.post('/recommend/interaction', data),
  getHistory: (params) => api.get('/recommend/history', { params }),
  getStats: () => api.get('/recommend/stats'),
  getMovieDetails: (id) => api.get(`/recommend/movie/${id}`),
};

export const searchService = {
  searchMovies: (params) => api.get('/search/movies', { params }),
  getGenres: () => api.get('/search/genres'),
};

export const watchlistService = {
  getWatchlist: () => api.get('/watchlist'),
  addToWatchlist: (movieId) => api.post('/watchlist', { movieId }),
  removeFromWatchlist: (movieId) => api.delete(`/watchlist/${movieId}`),
  checkWatchlist: (movieIds) => api.get('/watchlist/check', { params: { movieIds: movieIds.join(',') } }),
};

export const reviewService = {
  createReview: (data) => api.post('/reviews', data),
  getMovieReviews: (movieId, params) => api.get(`/reviews/movie/${movieId}`, { params }),
  getUserReviews: () => api.get('/reviews/user'),
  likeReview: (reviewId) => api.put(`/reviews/${reviewId}/like`),
  deleteReview: (reviewId) => api.delete(`/reviews/${reviewId}`),
};

export const socialService = {
  getFriends: () => api.get('/social/friends'),
  getPendingRequests: () => api.get('/social/pending'),
  getFriendActivity: () => api.get('/social/activity'),
  searchUsers: (q) => api.get('/social/search', { params: { q } }),
  sendFriendRequest: (recipientId) => api.post('/social/request', { recipientId }),
  respondFriendRequest: (friendshipId, action) => api.put(`/social/respond/${friendshipId}`, { action }),
  removeFriend: (friendshipId) => api.delete(`/social/friend/${friendshipId}`),
};

export const analyticsService = {
  getDashboardStats: () => api.get('/analytics/dashboard'),
  getCTRTrend: (days) => api.get('/analytics/ctr', { params: { days } }),
  getUserFunnel: () => api.get('/analytics/funnel'),
};

export const sessionService = {
  startSession: () => api.post('/sessions/start'),
  recordAction: (sessionId, data) => api.post(`/sessions/${sessionId}/action`, data),
  endSession: (sessionId) => api.post(`/sessions/${sessionId}/end`),
  getSessionReplay: (sessionId) => api.get(`/sessions/${sessionId}/replay`),
  getUserSessions: (params) => api.get('/sessions', { params }),
};

export default api;

