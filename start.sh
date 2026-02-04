#!/bin/bash
# ==============================================================================
# POW Agent - Startup Script for Render Background Worker
# ==============================================================================

set -e

echo "=========================================="
echo "  PROOF-OF-WORK AGENT - STARTING"
echo "=========================================="
echo "Environment: ${ENVIRONMENT:-production}"
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=========================================="

# Create necessary directories
mkdir -p logs data

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

# Run the agent daemon
exec python -u agent/main.py
