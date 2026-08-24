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
python scripts/train_mamba.py --run-name mamba_run1 --epochs 8
```

**The `--epochs 8` override is required to reproduce the specific
`mamba_run1` result reported in `paper/experimental_results.md`.**
`configs/default.yaml`'s `training.epochs` default is 30, not 8 — the
actual `mamba_run1` run used 8 epochs (see
`experiments/results/mamba_run1/config.json`, and
`paper/methodology_notes.md` §7 for why: the sequential-scan Mamba
implementation's per-epoch cost on this hardware made 8 epochs the
practical choice for this first run). Running the command without
`--epochs 8` will train for 30 epochs against the config default and
will NOT reproduce the reported 94.51% test accuracy / 77.9-minute
training time exactly — it will produce a different (longer) run.

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
python scripts/train_gru.py --run-name gru_run1 --epochs 8
python scripts/evaluate.py --run-name gru_run1
```

The same `--epochs 8` override applies here (see step 6's note) — the
actual `gru_run1` run used 8 epochs, not the config default of 30 (see
`experiments/results/gru_run1/config.json`). Identical protocol to
steps 6-7 (same manifests, same optimizer/schedule/early-stopping
settings, same epoch count), so the Mamba-vs-GRU comparison is fair.

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
measuring real forward+backward and inference timing, **plus real memory
usage**: process peak RSS (`resource.getrusage`, before/after each row)
and, on MPS rows, the PyTorch/Metal allocator counters
(`torch.mps.current_allocated_memory()` / `driver_allocated_memory()`).
Writes `experiments/results/benchmark_full.json`. See
`paper/experimental_results.md`'s "Full benchmark sweep — memory" section
for the measured figures and the methodology caveats (in particular: RSS
deltas are cumulative within one process, so the first row absorbs
one-time PyTorch/MPS import cost — MPS driver-allocated memory is the
cleaner per-row signal on MPS).

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

**Version control:** the project is under version control — `git init`
plus an initial commit (`f28fb63`, "Initial commit: Mamba speech-command
recognition project") on branch `main`, tracking `origin/main`. This
document previously (through the commit tagged below) stated the project
was "not yet under version control" / that `scripts/env_report.py`
reported `NOT_A_GIT_REPO_OR_GIT_UNAVAILABLE` — that statement is now
obsolete and has been removed; `env_report.py` runs from this point
onward record a real commit hash.

**Current benchmark/results commit:** `26bc1639d027a3cc697d3ebf2cb4eae27d233e99`
("Add real memory measurement (RSS + MPS allocator) to benchmark.py") is
the commit that added memory instrumentation to `scripts/benchmark.py`
and the corresponding measured output at
`experiments/results/benchmark_full.json`. This is the authoritative
commit for every latency and memory figure currently reported in
`paper/experimental_results.md` — `scripts/benchmark.py` and
`experiments/results/benchmark_full.json` as they exist at this commit
reproduce those numbers exactly (re-running the script will produce a
new run with ordinary hardware/timing variance, not necessarily
bit-identical figures, since the benchmark measures live process/GPU
memory and wall-clock timing rather than a seeded, purely deterministic
computation).

**Memory measurement:** `scripts/benchmark.py` was extended with real
memory instrumentation — process peak RSS via `resource.getrusage`
(`RUSAGE_SELF`, before/after each row) and, on MPS rows, the PyTorch/
Metal allocator counters `torch.mps.current_allocated_memory()` and
`torch.mps.driver_allocated_memory()` — and re-run on the target Mac at
commit `26bc1639d027a3cc697d3ebf2cb4eae27d233e99`. This closed the
previously-`NOT YET MEASURED` memory-usage rows in Experiments 1-2 of
`paper/experimental_results.md`. Three distinct quantities are recorded
per row and must not be conflated with each other or referred to
generically as "model memory":

- **Process peak RSS** (`process_delta_rss_mb`) — whole-process resident
  set size, cumulative within a single `benchmark.py` run (see the
  caveat in `paper/experimental_results.md` about the first row
  absorbing one-time PyTorch/MPS import cost).
- **MPS current-allocated memory** (`mps_current_allocated_mb`,
  `torch.mps.current_allocated_memory()`) — per PyTorch's own docstring,
  "the current GPU memory occupied by tensors," which explicitly
  **excludes** cached allocations held in MPSAllocator's memory pools.
  It stayed nearly flat across batch sizes in this benchmark and is not
  the informative per-row signal here.
- **MPS driver-allocated memory** (`mps_driver_allocated_mb`,
  `torch.mps.driver_allocated_memory()`) — per PyTorch's own docstring,
  "total GPU memory allocated by Metal driver for the process," which
  explicitly **includes** cached allocations in MPSAllocator pools as
  well as allocations from the MPS/MPSGraph frameworks. This is a
  process-wide driver-level figure, not a measurement isolated to the
  model's own tensors, and it is the more informative per-row MPS
  figure here since it scales cleanly with batch size and model.

See `paper/experimental_results.md`'s "Full benchmark sweep — memory"
section for the full measured table and the CPU batch=64 process-RSS
anomaly, which is reported as an observed result without an invented
causal explanation.

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
