# Reproducibility

Exact commands to reproduce every experiment in this project, in order.
Run from the project root with the virtual environment activated
(`source .venv/bin/activate`).

## 0. Environment setup

```bash
bash scripts/setup_env.sh
```

Creates `.venv`, installs `requirements.txt`, and runs the environment
report (prints Python/PyTorch/macOS/MPS details, executes a real MPS
tensor op). Re-run the report standalone anytime with:

```bash
python scripts/env_report.py --json experiments/logs/env_report.json
```

## 1. Dataset

```bash
python scripts/download_dataset.py     # downloads speech_commands_v0.02.tar.gz, verifies SHA256
python scripts/prepare_dataset.py       # builds train/val/test manifests, speaker-disjoint split
```

Manifests are written to `data/processed/{train,val,test}.csv`. The script
prints per-class sample counts and a speaker-leakage PASS/FAIL check.

## 2. Feature extraction sanity check

```bash
python scripts/visualize_features.py --n 8
```

Writes waveform + log-Mel spectrogram figures to
`experiments/figures/feature_samples/` and asserts all outputs are finite.

## 3. Unit tests (run before any training)

```bash
python -m pytest tests/ -v
```

Covers: audio I/O, feature extraction, Mamba (shape/gradients/causality/
CPU/MPS), GRU, and the shared classifier head.

## 4. Pre-training timing probe

```bash
python scripts/benchmark.py --probe-only
```

Measures real forward+backward and inference timing on this specific
machine (MPS if available, else CPU) at a small batch size, and estimates
full-epoch time using the actual prepared dataset size. Used to size
epoch count / batch size for the full training runs below to something
that completes in a practical amount of time on this hardware — see
`paper/methodology_notes.md` §7 for why this matters (the sequential-scan
Mamba implementation has real backward-pass overhead not present in the
GRU baseline).

## 5. Tiny-overfit sanity check (must pass before full training)

```bash
python scripts/tiny_overfit_test.py --backbone mamba
python scripts/tiny_overfit_test.py --backbone gru
```

Trains on a small class-balanced subset for many epochs; should reach
>=95% train accuracy on both backbones. If not, do not proceed to full
training — something in the pipeline is broken.

## 6. Full training — Mamba

```bash
python scripts/train_mamba.py --run-name mamba_run1
```

Optional overrides: `--epochs N --batch-size N`. Saves checkpoint to
`models/mamba_run1.pt`, training curves to
`experiments/results/mamba_run1/training_curves.{csv,json}`, and a run
summary (parameter count, device, total training time, final/best
accuracy) to `experiments/results/mamba_run1/summary.json`.

## 7. Evaluation — Mamba

```bash
python scripts/evaluate.py --run-name mamba_run1
```

Runs the checkpoint on `data/processed/test.csv`, writes
`experiments/results/mamba_run1/test_metrics.json` (accuracy, precision,
recall, F1, confusion matrix — all computed from real predictions), a
confusion-matrix figure, a training-curves figure, and appends a row to
`experiments/results/all_runs_metrics.csv`.

## 8. Full training + evaluation — GRU baseline

```bash
python scripts/train_gru.py --run-name gru_run1
python scripts/evaluate.py --run-name gru_run1
```

Identical protocol to steps 6-7 (same manifests, same optimizer/schedule/
early-stopping settings), so the Mamba-vs-GRU comparison is fair.

## 9. Noise robustness experiment

```bash
python scripts/noise_experiment.py --run-name mamba_run1
python scripts/noise_experiment.py --run-name gru_run1
python scripts/generate_comparison_figures.py --mamba-run mamba_run1 --gru-run gru_run1
```

(Also available as `bash scripts/checkpoint_9_noise.sh`.) Evaluates each
trained checkpoint against the test set under the noise conditions defined
in `configs/default.yaml: noise.conditions` (clean / mild / moderate /
strong, additive white Gaussian noise at specified SNRs), using a fixed
seed for reproducibility, then generates the Mamba-vs-GRU comparison
figures (accuracy, training time, parameter count, noise robustness) in
`experiments/figures/`.

## 10. Full benchmark (paper-ready latency/memory table)

```bash
python scripts/benchmark.py
```

Sweeps both backbones × both available devices × several batch sizes,
measuring real forward+backward and inference timing. Writes
`experiments/results/benchmark_full.json`.

## 11. Real-time microphone demo

```bash
python scripts/mic_check.py                          # optional smoke test first
python scripts/realtime_demo.py --run-name mamba_run1
```

Requires microphone permission on macOS (System Settings > Privacy &
Security > Microphone, granted to Terminal/your IDE on first run).
`mic_check.py` records 2 seconds and reports mean energy/peak amplitude as
a standalone check before launching the full visualization window. The
demo triggers on energy-based voice activity detection
(`realtime.vad_energy_threshold` in `configs/default.yaml`); each
prediction's real preprocessing/inference/total latency and softmax
probabilities are printed to the terminal and drawn in the plot window.

## STEP 12 — Final validation (clean-environment reproducibility check)

**Performed 2026-08-24 on the target Mac.** `.venv` was fully deleted and
rebuilt from scratch (`rm -rf .venv && bash scripts/setup_env.sh`), then:

- Full test suite: 40/40 PASS (identical to the pre-rebuild run).
- Existing checkpoints (`mamba_run1`, `gru_run1`) reloaded and
  re-evaluated on `data/processed/test.csv` without retraining:
  Mamba reproduced test accuracy 0.9451 / macro P-R-F1 0.9455-0.9450-0.9450;
  GRU reproduced test accuracy 0.9399 / macro P-R-F1 0.9400-0.9396-0.9395 —
  exact match to the originally recorded values in Experiments 1-2.
- Full training runs were NOT repeated from scratch in this validation
  pass (would cost ~78 min for Mamba alone); checkpoint-reload evaluation
  was judged sufficient to confirm environment/pipeline reproducibility
  without re-spending that time, since the recorded metrics came from a
  deterministic evaluation pass over a fixed checkpoint and fixed test
  set, not from a step with run-to-run randomness.

**Known reproducibility gap:** `scripts/env_report.py` reports
`git commit: NOT_A_GIT_REPO_OR_GIT_UNAVAILABLE` — this project has not
yet been placed under version control, so no result in this document is
tied to a specific commit hash. Recommended before the paper is finalized:
`git init` + an initial commit, so `env_report.py`'s git-commit field
becomes meaningful for future runs.

## Environment used for the reported results

See `experiments/logs/env_report.json` for the exact Python/PyTorch/macOS/
MPS/hardware configuration used to produce the numbers in
`paper/experimental_results.md`. That file is generated fresh by
`scripts/env_report.py` and is not hand-edited.

## Random seeds

All scripts read `project.seed` from `configs/default.yaml` (default: 42)
and call `torch.manual_seed(seed)` before model construction / training.
The noise-robustness experiment additionally uses a fixed, index-derived
seed per sample (see `src/training/dataset.py`) so the exact same noisy
audio is generated on every run.
