#!/bin/bash
# Script to run all code quality checks

set -e

echo "🔍 Running code quality checks..."

# Install dev dependencies if needed
echo "📦 Installing development dependencies..."
uv sync --extra dev

# Run the quality check script
echo "🚀 Running quality checks..."
uv run python scripts/check.py