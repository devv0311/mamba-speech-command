#!/usr/bin/env bash
# STEP 9 — Noise robustness experiment for both trained models, plus
# generates all Mamba-vs-GRU comparison figures (accuracy, training time,
# params, noise robustness) from the real saved results.
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate

echo "############################################"
echo "# STEP 9 — Noise robustness (Mamba)"
echo "############################################"
python scripts/noise_experiment.py --run-name mamba_run1

echo ""
echo "############################################"
echo "# STEP 9 — Noise robustness (GRU)"
echo "############################################"
python scripts/noise_experiment.py --run-name gru_run1

echo ""
echo "############################################"
echo "# Generate comparison figures"
echo "############################################"
python scripts/generate_comparison_figures.py --mamba-run mamba_run1 --gru-run gru_run1

echo ""
echo "== checkpoint_9_noise.sh complete =="
echo "Figures in experiments/figures/"
