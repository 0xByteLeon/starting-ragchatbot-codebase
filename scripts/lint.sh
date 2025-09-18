#!/bin/bash

# Lint Python code with Ruff

echo "🔍 Linting Python code with Ruff..."
uv run ruff check . --fix
echo "✅ Linting complete!"