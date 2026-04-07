#!/bin/bash
# Aider CLI integration test against rapid-mlx server.
#
# What it tests:
#   1. Aider can connect to the OpenAI-compatible endpoint
#   2. Aider can read a file, modify it via the LLM, and write the change back
#
# Pass = exit 0 AND the toy file contains the expected modification.
set -e

WORKDIR=$(mktemp -d)
trap "rm -rf $WORKDIR" EXIT
cd "$WORKDIR"

# Set up a tiny git repo (Aider requires git)
git init -q
git config user.email "test@test.local"
git config user.name "Test"

# Toy file: a Python function that needs a docstring added
cat > calc.py <<'EOF'
def add(a, b):
    return a + b
EOF
git add calc.py
git commit -q -m "init"

# Run aider with rapid-mlx as the backend.
# - openai/<model> tells LiteLLM to use the OpenAI-compatible client
# - --yes-always: don't prompt for confirmation
# - --no-auto-commits: don't pollute git
# - --no-show-model-warnings: skip the unknown-model warning
# - --no-stream: easier to debug, less flake on slow models
# - --map-tokens 0: don't waste a turn building a repo map
# - --message: one-shot mode, exits after the message
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=not-needed

aider \
    --model "openai//Volumes/Extreme SSD/mlx-models/gemma-4-26b-a4b-it-4bit" \
    --yes-always \
    --no-auto-commits \
    --no-show-model-warnings \
    --no-stream \
    --no-pretty \
    --map-tokens 0 \
    --no-check-update \
    --no-analytics \
    --message "Add a single-line docstring to the add function: \"Return the sum of a and b.\"" \
    calc.py 2>&1 | tail -40

echo
echo "=== Final calc.py ==="
cat calc.py
echo "=== END ==="

# Verify the file was modified
if grep -q '"""' calc.py || grep -q "'''" calc.py; then
    echo "PASS: docstring added"
    exit 0
else
    echo "FAIL: no docstring found in calc.py"
    exit 1
fi
