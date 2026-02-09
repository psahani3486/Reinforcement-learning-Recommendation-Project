# Use official Node.js runtime as base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy backend package files
COPY backend/package*.json ./

# Install dependencies
RUN npm install --production

# Copy the backend source code
COPY backend/src ./src

# Expose the port
EXPOSE 5000

# Start the server
CMD ["node", "src/server.js"]
