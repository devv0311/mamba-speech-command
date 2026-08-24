#!/usr/bin/env bash
# Pre-flight timing probe + STEP 5 (tiny overfit sanity check, both backbones).
# Run this BEFORE full training (STEP 6/8) — the probe gives a real per-batch
# timing measurement on your actual Mac (MPS + CPU), which is used to pick a
# sane epoch count/batch size for the full training run rather than guessing.
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate

echo "############################################"
echo "# Pre-flight timing probe (real hardware measurement, no full epoch)"
echo "############################################"
python scripts/benchmark.py --probe-only

echo ""
echo "############################################"
echo "# STEP 5 — Tiny overfit sanity check (Mamba)"
echo "############################################"
python scripts/tiny_overfit_test.py --backbone mamba

echo ""
echo "############################################"
echo "# STEP 5 — Tiny overfit sanity check (GRU)"
echo "############################################"
python scripts/tiny_overfit_test.py --backbone gru

echo ""
echo "== checkpoint_5_probe.sh complete =="
