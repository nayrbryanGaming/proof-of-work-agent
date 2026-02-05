#!/bin/bash
# ==============================================================================
# POW Agent - Startup Script for Render Background Worker
# ==============================================================================
# This script is designed to run on Render.com as a Background Worker.
# It handles all initialization and runs the agent daemon.
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          PROOF-OF-WORK AGENT - STARTUP                       ║"
echo "║          Colosseum Solana Agent Hackathon                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Environment: ${ENVIRONMENT:-production}"
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Python: $(python --version 2>&1)"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p logs data

# Validate required environment variables
echo "Validating environment..."
ERRORS=0

if [ -z "$COLOSSEUM_API_KEY" ]; then
    echo -e "${RED}ERROR: COLOSSEUM_API_KEY not set${NC}"
    ERRORS=$((ERRORS + 1))
fi

if [ -z "$AGENTWALLET_SESSION" ]; then
    echo -e "${RED}ERROR: AGENTWALLET_SESSION not set${NC}"
    ERRORS=$((ERRORS + 1))
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}WARNING: OPENAI_API_KEY not set (using fallback responses)${NC}"
fi

if [ -z "$PROGRAM_ID" ]; then
    echo -e "${YELLOW}WARNING: PROGRAM_ID not set (running in TEST MODE)${NC}"
fi

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}Found $ERRORS critical errors. Cannot start.${NC}"
    exit 1
fi

echo -e "${GREEN}Environment validated successfully${NC}"
echo ""

# Set Python environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1
export TZ=UTC

# Health check function
health_check() {
    echo "Running pre-flight health check..."
    python -c "
import sys
sys.path.insert(0, '.')
from agent.config import config
from agent.logger import get_logger
print('Config loaded successfully')
print(f'Loop interval: {config.agent.loop_interval}s')
" || {
    echo -e "${RED}Health check failed${NC}"
    exit 1
}
    echo -e "${GREEN}Health check passed${NC}"
}

# Run health check
health_check

echo ""
echo "=========================================="
echo "  Starting Agent Daemon"
echo "=========================================="
echo ""

# Run the agent daemon with proper signal handling
exec python -u agent/main.py

