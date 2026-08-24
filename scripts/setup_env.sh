#!/usr/bin/env bash
# One-time environment setup for macOS / Apple Silicon.
# Run this from the project root:
#   bash scripts/setup_env.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "== Project root: $PROJECT_ROOT =="

if [ ! -d ".venv" ]; then
  echo "== Creating virtual environment (.venv) =="
  python3 -m venv .venv
else
  echo "== .venv already exists, reusing it =="
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "== Upgrading pip =="
pip install --upgrade pip

echo "== Installing requirements =="
pip install -r requirements.txt

echo "== Verifying environment (Python / PyTorch / MPS) =="
python scripts/env_report.py --json experiments/logs/env_report.json

echo ""
echo "== Setup complete. Activate with: source .venv/bin/activate =="
