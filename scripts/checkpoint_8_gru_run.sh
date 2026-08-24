#!/usr/bin/env bash
# STEP 8 — GRU baseline, same protocol as the Mamba run (mamba_run1): same
# dataset manifests, same 8 epochs, same optimizer/schedule — for a fair,
# apples-to-apples comparison. Based on the batch=64 probe, GRU is much
# faster (~0.5 min/epoch on MPS), so this should complete in a few minutes,
# not over an hour like the Mamba run.
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate

echo "############################################"
echo "# STEP 8 — GRU baseline training (8 epochs, matched to mamba_run1)"
echo "############################################"
python scripts/train_gru.py --run-name gru_run1 --epochs 8

echo ""
echo "############################################"
echo "# STEP 8 — GRU evaluation on held-out test set"
echo "############################################"
python scripts/evaluate.py --run-name gru_run1

echo ""
echo "== checkpoint_8_gru_run.sh complete =="
