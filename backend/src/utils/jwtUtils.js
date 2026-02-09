/**
 * JWT Utilities - Helper functions for token generation and verification
 */

import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'your-super-secret-jwt-key-change-this';
const JWT_EXPIRY = '30d';

/**
 * Generate JWT token for a user
 * @param {string} userId - User ID from MongoDB
 * @returns {string} JWT token
 */
export const generateToken = (userId) => {
  return jwt.sign({ id: userId }, JWT_SECRET, {
    expiresIn: JWT_EXPIRY,
  });
};

/**
 * Verify JWT token and extract payload
 * @param {string} token - JWT token
 * @returns {object} Decoded payload with userId
 * @throws {Error} If token is invalid or expired
 */
export const verifyToken = (token) => {
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    return decoded;
  } catch (error) {
    throw new Error(`Invalid token: ${error.message}`);
  }
};

/**
 * Decode token without verification (for debugging only)
 * @param {string} token - JWT token
 * @returns {object} Decoded payload
 */
export const decodeToken = (token) => {
  return jwt.decode(token);
};

/**
 * Check if token is expired
 * @param {string} token - JWT token
 * @returns {boolean} True if expired, false otherwise
 */
export const isTokenExpired = (token) => {
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    const expiryTime = decoded.exp * 1000; // Convert to milliseconds
    return expiryTime < Date.now();
  } catch (error) {
    return true; // If verification fails, token is expired/invalid
  }
};

/**
 * Get time until token expires
 * @param {string} token - JWT token
 * @returns {number} Milliseconds until expiry, or 0 if expired
 */
export const getTokenExpiryTime = (token) => {
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    const expiryTime = decoded.exp * 1000;
    const timeLeft = expiryTime - Date.now();
    return timeLeft > 0 ? timeLeft : 0;
  } catch (error) {
    return 0;
  }
};

/**
 * Refresh token by generating a new one
 * @param {string} oldToken - Old JWT token
 * @returns {string} New JWT token
 */
export const refreshToken = (oldToken) => {
  try {
    const decoded = verifyToken(oldToken);
    return generateToken(decoded.id);
  } catch (error) {
    throw new Error('Cannot refresh invalid token');
  }
};

export default {
  generateToken,
  verifyToken,
  decodeToken,
  isTokenExpired,
  getTokenExpiryTime,
  refreshToken,
};
