#!/usr/bin/env bash
# Environment setup script for narrative-learning
set -euo pipefail

# Update package lists and install PostgreSQL
sudo apt-get update
sudo apt-get install -y postgresql postgresql-client

# Install uv for managing Python packages
if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install Python dependencies with uv
uv pip install -r requirements.txt

echo "Environment setup complete."
