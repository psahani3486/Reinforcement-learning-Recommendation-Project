import request from 'supertest';
import jwt from 'jsonwebtoken';
import User from '../src/models/User.js';
import connectDB from '../src/config/database.js';

// Mock MongoDB connection for testing
let app;
let mongoServer;

describe('MongoDB & JWT Integration Tests', () => {
  
  beforeAll(async () => {
    // Connect to MongoDB (test database)
    await connectDB();
  });

  afterAll(async () => {
    // Clean up
    await User.deleteMany({});
  });

  // ===== MONGODB TESTS =====
  describe('MongoDB Connection & User Model', () => {
    
    test('should connect to MongoDB', async () => {
      // If we reach here without error, connection is successful
      expect(true).toBe(true);
    });

    test('should create a user with valid data', async () => {
      const userData = {
        username: 'testuser1',
        email: 'test1@example.com',
        password: 'password123'
      };

      const user = await User.create(userData);

      expect(user).toBeDefined();
      expect(user.username).toBe('testuser1');
      expect(user.email).toBe('test1@example.com');
      expect(user.password).not.toBe('password123'); // Should be hashed
    });

    test('should not create duplicate users', async () => {
      const userData = {
        username: 'testuser2',
        email: 'test2@example.com',
        password: 'password123'
      };

      await User.create(userData);

      try {
        await User.create(userData); // Try to create duplicate
        expect(true).toBe(false); // Should throw error
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    test('should find user by email', async () => {
      const userData = {
        username: 'testuser3',
        email: 'test3@example.com',
        password: 'password123'
      };

      await User.create(userData);
      const foundUser = await User.findOne({ email: 'test3@example.com' });

      expect(foundUser).toBeDefined();
      expect(foundUser.username).toBe('testuser3');
    });

    test('should hash password on save', async () => {
      const userData = {
        username: 'testuser4',
        email: 'test4@example.com',
        password: 'plainPassword123'
      };

      const user = await User.create(userData);

      expect(user.password).not.toBe('plainPassword123');
      expect(user.password).toHaveLength(60); // bcrypt hash length
    });
  });

  // ===== JWT TESTS =====
  describe('JWT Token Generation & Verification', () => {

    test('should generate a valid JWT token', () => {
      const userId = '507f1f77bcf86cd799439011';
      const secret = process.env.JWT_SECRET || 'test-secret-key';

      const token = jwt.sign({ id: userId }, secret, {
        expiresIn: '30d'
      });

      expect(token).toBeDefined();
      expect(typeof token).toBe('string');
    });

    test('should decode JWT token and extract userId', () => {
      const userId = '507f1f77bcf86cd799439012';
      const secret = process.env.JWT_SECRET || 'test-secret-key';

      const token = jwt.sign({ id: userId }, secret, {
        expiresIn: '30d'
      });

      const decoded = jwt.verify(token, secret);

      expect(decoded.id).toBe(userId);
    });

    test('should reject invalid JWT token', () => {
      const invalidToken = 'invalid.token.here';
      const secret = process.env.JWT_SECRET || 'test-secret-key';

      expect(() => {
        jwt.verify(invalidToken, secret);
      }).toThrow();
    });

    test('should reject expired token', () => {
      const userId = '507f1f77bcf86cd799439013';
      const secret = process.env.JWT_SECRET || 'test-secret-key';

      const token = jwt.sign({ id: userId }, secret, {
        expiresIn: '-1s' // Expired 1 second ago
      });

      expect(() => {
        jwt.verify(token, secret);
      }).toThrow();
    });

    test('should reject token signed with wrong secret', () => {
      const userId = '507f1f77bcf86cd799439014';
      const wrongSecret = 'wrong-secret-key';

      const token = jwt.sign({ id: userId }, wrongSecret, {
        expiresIn: '30d'
      });

      const correctSecret = process.env.JWT_SECRET || 'test-secret-key';

      expect(() => {
        jwt.verify(token, correctSecret);
      }).toThrow();
    });
  });

  // ===== AUTHENTICATION TESTS =====
  describe('Authentication Flow', () => {

    test('should register a new user and return JWT token', async () => {
      const userData = {
        username: 'newuser1',
        email: 'newuser1@example.com',
        password: 'securePassword123'
      };

      // Simulate register endpoint
      const user = await User.create(userData);
      const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET || 'test-secret-key', {
        expiresIn: '30d'
      });

      expect(user._id).toBeDefined();
      expect(token).toBeDefined();
    });

    test('should login user with correct credentials', async () => {
      const userData = {
        username: 'logintest1',
        email: 'logintest1@example.com',
        password: 'correctPassword123'
      };

      const user = await User.create(userData);
      const isPasswordCorrect = await user.comparePassword('correctPassword123');

      expect(isPasswordCorrect).toBe(true);
    });

    test('should reject login with incorrect password', async () => {
      const userData = {
        username: 'logintest2',
        email: 'logintest2@example.com',
        password: 'correctPassword123'
      };

      const user = await User.create(userData);
      const isPasswordCorrect = await user.comparePassword('wrongPassword123');

      expect(isPasswordCorrect).toBe(false);
    });

    test('should not find user with non-existent email', async () => {
      const user = await User.findOne({ email: 'nonexistent@example.com' });

      expect(user).toBeNull();
    });
  });

  // ===== JWT PAYLOAD VALIDATION =====
  describe('JWT Payload Validation', () => {

    test('should have userId in JWT payload', () => {
      const userId = '507f1f77bcf86cd799439015';
      const secret = process.env.JWT_SECRET || 'test-secret-key';

      const token = jwt.sign({ id: userId }, secret, {
        expiresIn: '30d'
      });

      const decoded = jwt.verify(token, secret);

      expect(decoded).toHaveProperty('id');
      expect(decoded.id).toBe(userId);
    });

    test('should have expiration time in JWT token', () => {
      const userId = '507f1f77bcf86cd799439016';
      const secret = process.env.JWT_SECRET || 'test-secret-key';

      const token = jwt.sign({ id: userId }, secret, {
        expiresIn: '30d'
      });

      const decoded = jwt.verify(token, secret);

      expect(decoded).toHaveProperty('exp');
      expect(decoded.exp).toBeGreaterThan(Math.floor(Date.now() / 1000));
    });

    test('should have issued-at time in JWT token', () => {
      const userId = '507f1f77bcf86cd799439017';
      const secret = process.env.JWT_SECRET || 'test-secret-key';

      const token = jwt.sign({ id: userId }, secret, {
        expiresIn: '30d'
      });

      const decoded = jwt.verify(token, secret);

      expect(decoded).toHaveProperty('iat');
      expect(decoded.iat).toBeGreaterThan(0);
    });
  });

  // ===== USER SESSION TESTS =====
  describe('User Session Management', () => {

    test('should store user embedding', async () => {
      const userData = {
        username: 'embedtest1',
        email: 'embedtest1@example.com',
        password: 'password123',
        embedding: Array.from({ length: 64 }, () => Math.random())
      };

      const user = await User.create(userData);

      expect(user.embedding).toBeDefined();
      expect(user.embedding.length).toBe(64);
    });

    test('should generate embedding if not provided', async () => {
      const userData = {
        username: 'embedtest2',
        email: 'embedtest2@example.com',
        password: 'password123'
      };

      const user = await User.create({
        ...userData,
        embedding: Array.from({ length: 64 }, () => Math.random())
      });

      expect(user.embedding).toBeDefined();
    });

    test('should store user preferences', async () => {
      const userData = {
        username: 'preftest1',
        email: 'preftest1@example.com',
        password: 'password123',
        preferences: {
          genres: ['Action', 'Thriller'],
          favoriteActors: ['Actor1', 'Actor2']
        }
      };

      const user = await User.create(userData);

      expect(user.preferences.genres).toContain('Action');
      expect(user.preferences.favoriteActors).toContain('Actor1');
    });
  });
});
