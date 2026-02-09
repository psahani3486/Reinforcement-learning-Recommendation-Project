export default {
  testEnvironment: 'node',
  transform: {},
  testMatch: ['**/tests/**/*.test.js'],
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/server.js',
    '!node_modules/**'
  ],
  coveragePathIgnorePatterns: [
    '/node_modules/',
    '/tests/'
  ],
  testTimeout: 30000,
  forceExit: true,
  clearMocks: true,
  resetMocks: true,
  setupFilesAfterEnv: ['<rootDir>/tests/setup.js']
};
