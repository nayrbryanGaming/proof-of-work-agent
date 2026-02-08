#!/usr/bin/env bash
set -euo pipefail

# Helper to install a GitHub Actions self-hosted runner for
# https://github.com/nayrbryanGaming/proof-of-work-agent
# Usage: ./setup_selfhosted_runner.sh <RUNNER_TOKEN> [RUNNER_NAME]

REPO_URL="https://github.com/nayrbryanGaming/proof-of-work-agent"
TOKEN="${1:-}"
RUNNER_NAME="${2:-self-hosted-runner}"

if [ -z "$TOKEN" ]; then
  echo "Error: runner registration token missing."
  echo "Get a token from: ${REPO_URL}/settings/actions/runners/new"
  exit 1
fi

mkdir -p actions-runner
cd actions-runner

echo "Downloading latest Actions runner..."
ARCHIVE_URL="https://github.com/actions/runner/releases/latest/download/actions-runner-linux-x64.tar.gz"
curl -fsSLO "$ARCHIVE_URL"
tar xzf actions-runner-linux-x64.tar.gz

echo "Configuring runner for ${REPO_URL} (name=${RUNNER_NAME})"
./config.sh --url "$REPO_URL" --token "$TOKEN" --name "$RUNNER_NAME" --work _work

echo "Installing runner as a service and starting... (requires sudo)"
sudo ./svc.sh install
sudo ./svc.sh start

echo "Self-hosted runner installed and started."
echo "Check status with: sudo ./svc.sh status"
echo "To remove: sudo ./svc.sh stop && sudo ./svc.sh uninstall && ./config.sh remove --token <token>"
