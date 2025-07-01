#!/bin/bash
# WebUI Startup Script for Linux/macOS
# Use local project dependencies to avoid version conflicts

echo "🚀 Starting KrunchWrapper WebUI..."

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

# Use local vite to avoid version conflicts  
echo "⚡ Starting Vite development server..."
echo "🌐 WebUI will be available at: http://localhost:5173"

# Run with local vite from node_modules
./node_modules/.bin/vite --host 0.0.0.0 --port 5173 