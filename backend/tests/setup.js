/**
 * Jest Setup File
 * Runs before all tests
 */

// Increase test timeout for MongoDB operations
jest.setTimeout(30000);

// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.JWT_SECRET = 'test-jwt-secret-key-12345678901234567890';
process.env.MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/rl-recommendation-test';

console.log('Jest test environment initialized');
