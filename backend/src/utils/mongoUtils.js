/**
 * MongoDB Utilities - Helper functions for database operations
 */

import User from '../models/User.js';

/**
 * Create a new user with validation
 * @param {object} userData - { username, email, password }
 * @returns {object} Created user document
 * @throws {Error} If user already exists or validation fails
 */
export const createUserSafe = async (userData) => {
  const { username, email, password } = userData;

  if (!username || !email || !password) {
    throw new Error('Username, email, and password are required');
  }

  const existingUser = await User.findOne({
    $or: [{ email }, { username }],
  });

  if (existingUser) {
    const field = existingUser.email === email ? 'email' : 'username';
    throw new Error(`User with this ${field} already exists`);
  }

  const user = await User.create({
    username,
    email,
    password,
    embedding: Array.from({ length: 64 }, () => Math.random()),
  });

  return user;
};

/**
 * Find user by email
 * @param {string} email - User email
 * @returns {object|null} User document or null if not found
 */
export const findUserByEmail = async (email) => {
  return User.findOne({ email }).select('-password');
};

/**
 * Find user by username
 * @param {string} username - Username
 * @returns {object|null} User document or null if not found
 */
export const findUserByUsername = async (username) => {
  return User.findOne({ username }).select('-password');
};

/**
 * Find user by ID
 * @param {string} userId - MongoDB user ID
 * @returns {object|null} User document or null if not found
 */
export const findUserById = async (userId) => {
  return User.findById(userId).select('-password');
};

/**
 * Update user preferences
 * @param {string} userId - MongoDB user ID
 * @param {object} preferences - { genres, favoriteActors }
 * @returns {object} Updated user document
 */
export const updateUserPreferences = async (userId, preferences) => {
  const user = await User.findById(userId);

  if (!user) {
    throw new Error('User not found');
  }

  if (preferences.genres) {
    user.preferences.genres = preferences.genres;
  }

  if (preferences.favoriteActors) {
    user.preferences.favoriteActors = preferences.favoriteActors;
  }

  return user.save();
};

/**
 * Update user embedding (for ML recommendations)
 * @param {string} userId - MongoDB user ID
 * @param {array} embedding - New embedding vector
 * @returns {object} Updated user document
 */
export const updateUserEmbedding = async (userId, embedding) => {
  if (!Array.isArray(embedding)) {
    throw new Error('Embedding must be an array');
  }

  const user = await User.findById(userId);

  if (!user) {
    throw new Error('User not found');
  }

  user.embedding = embedding;
  return user.save();
};

/**
 * Delete user by ID (permanent)
 * @param {string} userId - MongoDB user ID
 * @returns {object} Deleted user document
 */
export const deleteUser = async (userId) => {
  const user = await User.findByIdAndDelete(userId);

  if (!user) {
    throw new Error('User not found');
  }

  return user;
};

/**
 * Count total users
 * @returns {number} Total user count
 */
export const getUserCount = async () => {
  return User.countDocuments();
};

/**
 * Get all users (admin only - be careful!)
 * @param {number} limit - Max users to return
 * @returns {array} User documents
 */
export const getAllUsers = async (limit = 100) => {
  return User.find().select('-password').limit(limit);
};

/**
 * Verify user password
 * @param {string} email - User email
 * @param {string} password - Plain text password
 * @returns {object|null} User object if password correct, null otherwise
 */
export const verifyUserPassword = async (email, password) => {
  const user = await User.findOne({ email });

  if (!user) {
    return null;
  }

  const isCorrect = await user.comparePassword(password);
  return isCorrect ? user : null;
};

export default {
  createUserSafe,
  findUserByEmail,
  findUserByUsername,
  findUserById,
  updateUserPreferences,
  updateUserEmbedding,
  deleteUser,
  getUserCount,
  getAllUsers,
  verifyUserPassword,
};
