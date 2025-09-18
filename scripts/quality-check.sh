#!/bin/bash

# Quality Check Script for RAG Chatbot
# Runs code formatting, linting, and tests

set -e

echo "🔧 Running Quality Checks..."

echo ""
echo "📝 Formatting code with Black..."
uv run black .

echo ""
echo "🔍 Linting with Ruff..."
uv run ruff check . --fix

echo ""
echo "🧪 Running tests..."
cd backend
uv run pytest tests/ -v

echo ""
echo "✅ All quality checks passed!"