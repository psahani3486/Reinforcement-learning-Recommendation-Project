# MongoDB & JWT Integration Tests

Complete integration test suite for MongoDB user management and JWT authentication.

## 📋 Overview

This test suite validates:
- **MongoDB Connection** - Database connectivity and CRUD operations
- **User Model** - Schema validation, password hashing, duplicates
- **JWT Security** - Token generation, verification, expiration, refresh
- **Authentication Flow** - Registration, login, session management
- **Error Handling** - Invalid tokens, expired tokens, unauthorized access

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd backend
npm install
```

### 2. Set Up Test Database

**Option A: Local MongoDB**
```bash
# Windows (if MongoDB installed)
mongod

# Or start MongoDB service
net start MongoDB
```

**Option B: MongoDB Atlas (Cloud)**
```bash
# Set environment variable for test database
set MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/rl-recommendation-test?retryWrites=true&w=majority
```

### 3. Run Tests
```bash
# Run all tests
npm test

# Run tests in watch mode (auto-rerun on file changes)
npm run test:watch

# Generate coverage report
npm run test:coverage
```

## 📊 Test Structure

### MongoDB Tests
```
✅ Connection
  - Should connect to MongoDB
  - Should create user with valid data
  - Should prevent duplicate users
  - Should find user by email
  - Should hash password on save

✅ User Model
  - Should validate required fields
  - Should enforce email format
  - Should enforce password minimum length
  - Should store preferences
  - Should store embeddings
```

### JWT Tests
```
✅ Token Generation
  - Should generate valid JWT token
  - Should include userId in token
  - Should set expiration time (30 days)
  - Should include issued-at timestamp

✅ Token Verification
  - Should decode valid token
  - Should extract userId from token
  - Should reject invalid token
  - Should reject expired token
  - Should reject wrong secret
  - Should reject malformed token

✅ Token Management
  - Should check token expiry
  - Should calculate time until expiry
  - Should refresh token
  - Should verify payload structure
```

### Authentication Tests
```
✅ Registration
  - Should create new user
  - Should return JWT token on registration
  - Should reject duplicate email
  - Should reject duplicate username
  - Should hash password

✅ Login
  - Should accept correct password
  - Should reject incorrect password
  - Should return JWT token on login
  - Should return user data
  - Should exclude password from response

✅ Session
  - Should maintain session with token
  - Should identify user from token
  - Should handle concurrent sessions
```

## 🧪 Example Tests

### Test 1: Create User
```javascript
test('should create a user with valid data', async () => {
  const userData = {
    username: 'testuser1',
    email: 'test1@example.com',
    password: 'password123'
  };

  const user = await User.create(userData);

  expect(user).toBeDefined();
  expect(user.username).toBe('testuser1');
  expect(user.password).not.toBe('password123'); // Hashed
});
```

### Test 2: JWT Verification
```javascript
test('should decode JWT token and extract userId', () => {
  const userId = '507f1f77bcf86cd799439012';
  const secret = process.env.JWT_SECRET;

  const token = jwt.sign({ id: userId }, secret, {
    expiresIn: '30d'
  });

  const decoded = jwt.verify(token, secret);

  expect(decoded.id).toBe(userId);
});
```

### Test 3: Login Flow
```javascript
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
```

## 🛠️ Utility Functions

### JWT Utilities (`src/utils/jwtUtils.js`)
```javascript
import { generateToken, verifyToken, isTokenExpired } from './jwtUtils.js';

// Generate token
const token = generateToken(userId);

// Verify token
const decoded = verifyToken(token); // { id, iat, exp }

// Check if expired
const expired = isTokenExpired(token);

// Get time remaining
const timeLeft = getTokenExpiryTime(token);

// Refresh token
const newToken = refreshToken(oldToken);
```

### MongoDB Utilities (`src/utils/mongoUtils.js`)
```javascript
import { 
  createUserSafe, 
  findUserByEmail, 
  updateUserPreferences 
} from './mongoUtils.js';

// Create user safely
const user = await createUserSafe({
  username: 'john',
  email: 'john@example.com',
  password: 'secret123'
});

// Find user
const user = await findUserByEmail('john@example.com');

// Update preferences
await updateUserPreferences(userId, {
  genres: ['Action', 'Thriller'],
  favoriteActors: ['Actor1']
});

// Verify password
const user = await verifyUserPassword(email, password);
```

## 📝 Test Output Example

```
PASS  tests/integration.test.js (8.234 s)
  MongoDB Connection & User Model
    ✓ should connect to MongoDB (45 ms)
    ✓ should create a user with valid data (125 ms)
    ✓ should not create duplicate users (89 ms)
    ✓ should find user by email (52 ms)
    ✓ should hash password on save (134 ms)
  JWT Token Generation & Verification
    ✓ should generate a valid JWT token (23 ms)
    ✓ should decode JWT token and extract userId (18 ms)
    ✓ should reject invalid JWT token (12 ms)
    ✓ should reject expired token (15 ms)
    ✓ should reject token signed with wrong secret (14 ms)
  Authentication Flow
    ✓ should register a new user and return JWT token (115 ms)
    ✓ should login user with correct credentials (98 ms)
    ✓ should reject login with incorrect password (89 ms)
    ✓ should not find user with non-existent email (34 ms)
  JWT Payload Validation
    ✓ should have userId in JWT payload (19 ms)
    ✓ should have expiration time in JWT token (16 ms)
    ✓ should have issued-at time in JWT token (15 ms)
  User Session Management
    ✓ should store user embedding (123 ms)
    ✓ should generate embedding if not provided (98 ms)
    ✓ should store user preferences (105 ms)

Test Suites: 1 passed, 1 total
Tests:       20 passed, 20 total
Snapshots:   0 total
Time:        8.234 s
```

## 🔧 Environment Variables

### Test Environment
```env
NODE_ENV=test
JWT_SECRET=test-jwt-secret-key-12345678901234567890
MONGODB_URI=mongodb://localhost:27017/rl-recommendation-test
```

### Production Environment
```env
NODE_ENV=production
JWT_SECRET=your-strong-secret-key-min-32-chars-here
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/rl-recommendation?retryWrites=true&w=majority
```

## 🚨 Troubleshooting

### MongoDB Connection Fails
```bash
# Check if MongoDB is running
mongod --version

# Start MongoDB
mongod
# or
net start MongoDB  # Windows
# or
systemctl start mongod  # Linux
```

### Tests Timeout
- Increase timeout in `jest.config.js`: `testTimeout: 60000`
- Check MongoDB Atlas network access (whitelist your IP)
- Verify `MONGODB_URI` is correct

### JWT Secret Mismatch
- Ensure `JWT_SECRET` env var is set
- Test uses: `test-jwt-secret-key-12345678901234567890`
- Production must use: strong random string (min 32 chars)

### Password Hashing Issues
- Verify `bcryptjs` is installed: `npm install bcryptjs`
- Password is hashed in User schema middleware
- Never compare plaintext to hash; use `user.comparePassword()`

## 📚 Related Files

- **Models**: `src/models/User.js`
- **Controllers**: `src/controllers/authController.js`
- **Middleware**: `src/middleware/auth.js`
- **Config**: `src/config/database.js`
- **Utilities**: 
  - `src/utils/jwtUtils.js` (JWT helpers)
  - `src/utils/mongoUtils.js` (MongoDB helpers)

## ✅ Testing Checklist

- [ ] MongoDB connection established
- [ ] All user CRUD operations work
- [ ] Password hashing verified
- [ ] JWT generation tested
- [ ] Token verification tested
- [ ] Expired token rejected
- [ ] Invalid token rejected
- [ ] User registration works
- [ ] User login works
- [ ] Preferences stored correctly
- [ ] Embeddings generated correctly
- [ ] Coverage > 80%

## 🎯 Next Steps

1. **Run tests**: `npm test`
2. **Fix any failures** (unlikely with pre-existing code)
3. **Deploy backend** to Railway
4. **Deploy frontend** to Vercel
5. **Integration test in production**

## 📖 Documentation

- JWT: https://jwt.io/
- MongoDB: https://docs.mongodb.com/
- Mongoose: https://mongoosejs.com/
- bcryptjs: https://github.com/dcodeIO/bcrypt.js
- Jest: https://jestjs.io/

---

**Happy testing! 🧪✨**
