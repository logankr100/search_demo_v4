#!/bin/bash

# Semantic Search Frontend - Start Script
# This script starts both the Flask backend and React frontend

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Semantic Search Frontend${NC}"
echo "========================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo "Please create a .env file with your OpenAI API key:"
    echo "  echo 'OPENAI_API_KEY=your_key_here' > .env"
    exit 1
fi

# Check if index_out_v2 exists
if [ ! -d index_out_v2 ]; then
    echo -e "${RED}Error: index_out_v2 directory not found${NC}"
    echo "Please run the data pipeline scripts to generate the search index"
    exit 1
fi

# Check if Python dependencies are installed
echo -e "${YELLOW}Checking Python dependencies...${NC}"
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip install -r api/requirements.txt
fi

# Check if Node dependencies are installed
if [ ! -d frontend/node_modules ]; then
    echo -e "${YELLOW}Installing Node dependencies...${NC}"
    cd frontend
    npm install
    cd ..
fi

echo -e "${GREEN}Starting Flask backend on port 5000...${NC}"
cd api
python3 server.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 2

echo -e "${GREEN}Starting React frontend on port 3000...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo -e "${GREEN}Frontend is starting up!${NC}"
echo "========================================"
echo "Backend:  http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}Servers stopped${NC}"
    exit 0
}

trap cleanup INT TERM

# Wait for either process to exit
wait
