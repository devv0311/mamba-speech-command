#!/usr/bin/env bash
# STEP 6 — First full Mamba training run, time-bounded to get real results
# quickly rather than committing to the full 30-epoch config blind.
#
# Based on your real MPS batch=64 timing probe (~9.4 min/epoch for Mamba),
# this runs 8 epochs (~75 min estimated, likely less due to early stopping
# with patience=6 if validation loss plateaus sooner). Once we see the real
# accuracy/loss trend from this run, we decide whether more epochs are
# worth the added time.
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate

echo "############################################"
echo "# STEP 6 — Mamba training (8 epochs, real run — this will take a while)"
echo "############################################"
python scripts/train_mamba.py --run-name mamba_run1 --epochs 8

echo ""
echo "############################################"
echo "# STEP 7 — Mamba evaluation on held-out test set"
echo "############################################"
python scripts/evaluate.py --run-name mamba_run1

echo ""
echo "== checkpoint_6_first_run.sh complete =="
echo "Real metrics: experiments/results/mamba_run1/test_metrics.json"
echo "Real training time: experiments/results/mamba_run1/summary.json"
