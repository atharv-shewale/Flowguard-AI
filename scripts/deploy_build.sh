#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python train.py
python save_bento_model.py
python -m bentoml build

echo "Build complete."
echo "To run Bento service:"
echo "  . .venv/bin/activate && bentoml serve service:DiabetesService --port \${BENTO_PORT:-3000}"
