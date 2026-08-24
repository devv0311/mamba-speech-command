#!/usr/bin/env bash
# Runs STEP 2 (dataset), STEP 3 (features), STEP 4 (Mamba/GRU tests incl. MPS)
# in one pass. Paste the full terminal output back for review.
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate

echo "############################################"
echo "# STEP 2 — Download dataset (~2.4GB, may take a few minutes)"
echo "############################################"
python scripts/download_dataset.py

echo ""
echo "############################################"
echo "# STEP 2 — Prepare 8-class manifest (speaker-disjoint split)"
echo "############################################"
python scripts/prepare_dataset.py

echo ""
echo "############################################"
echo "# STEP 3 — Feature extraction sample visualizations"
echo "############################################"
python scripts/visualize_features.py --n 8

echo ""
echo "############################################"
echo "# STEP 4 — Full test suite (audio, features, Mamba, GRU, classifier)"
echo "############################################"
python -m pytest tests/ -v

echo ""
echo "== checkpoint_2_3_4.sh complete =="
