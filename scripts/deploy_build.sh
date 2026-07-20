#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

if [[ ! "$PYTHON_VERSION" =~ ^3\.(9|10|11)$ ]]; then
  echo "Unsupported Python version: $PYTHON_VERSION"
  echo "Use Python 3.9, 3.10, or 3.11 for BentoML build compatibility."
  exit 1
fi

$PYTHON_BIN -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python train.py
python save_bento_model.py
python -m bentoml build

echo "Build complete."
echo "To run Bento service:"
echo "  . .venv/bin/activate && bentoml serve service:DiabetesService --port \${BENTO_PORT:-3000}"
