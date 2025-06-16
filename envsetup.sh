#!/usr/bin/env bash
# Environment setup script for narrative-learning
set -euo pipefail

# Update package lists and install PostgreSQL
sudo apt-get update
sudo apt-get install -y postgresql postgresql-client
curl -LsSf https://astral.sh/uv/install.sh | sh

uv run uvbootstrapper.py

