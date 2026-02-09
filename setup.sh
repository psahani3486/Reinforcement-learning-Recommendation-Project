#!/bin/bash

echo "🎬 RL Recommendation System - MERN Stack Setup"
echo "=============================================="
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

echo "✓ Node.js version: $(node -v)"

# Check if MongoDB is installed
if ! command -v mongod &> /dev/null; then
    echo "⚠️  MongoDB not found in PATH. Make sure MongoDB is installed and running."
else
    echo "✓ MongoDB found"
fi

echo ""
echo "📦 Installing Backend Dependencies..."
cd backend
npm install

if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️  Please update the .env file with your configuration"
fi

echo ""
echo "📦 Installing Frontend Dependencies..."
cd ../frontend
npm install

if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
fi

cd ..

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 Next Steps:"
echo "=============="
echo ""
echo "1. Make sure MongoDB is running:"
echo "   mongod"
echo ""
echo "2. Seed the database with sample movies (optional):"
echo "   cd backend && npm run seed"
echo ""
echo "3. Start the backend server:"
echo "   cd backend && npm run dev"
echo ""
echo "4. In a new terminal, start the frontend:"
echo "   cd frontend && npm run dev"
echo ""
echo "5. Open your browser and go to:"
echo "   http://localhost:3000"
echo ""
echo "✨ Happy coding!"
