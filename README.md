# Mamba Speech Command Recognition

Real-time, closed-vocabulary speech-command recognition using a compact
**Mamba (Selective State Space Model)**, implemented in PyTorch for
Apple Silicon (MPS), with a matched-capacity **GRU baseline** for
comparison.

**Research question:** Can a compact Mamba-based model provide accurate
and low-latency recognition of spoken commands on a resource-constrained
consumer device?

**Command vocabulary (8 classes):** yes, no, up, down, left, right, go, stop

**Target hardware:** MacBook Pro 14" (Apple M3 Pro, 11 CPU cores, 18 GB RAM),
PyTorch MPS backend, CPU fallback.

## Status

See `paper/experimental_results.md` for verified, measured results —
accuracy, latency, and memory usage for both Mamba and the GRU baseline,
noise robustness, and live-microphone real-time inference, all measured
on the target Mac. No numbers in this repository are fabricated; any
metric not yet run would be explicitly marked `NOT YET MEASURED`, and as
of commit `26bc1639d027a3cc697d3ebf2cb4eae27d233e99` no metric in the
project brief remains in that state.

## Quick start

```bash
git clone <this-repo>  # or just cd into it if already local
cd mamba-speech-command
bash scripts/setup_env.sh          # creates .venv, installs deps, runs env_report.py
source .venv/bin/activate
python scripts/env_report.py       # re-check environment anytime
```

See `paper/reproducibility.md` for the full command sequence to reproduce
every experiment.

## Project structure

```
mamba-speech-command/
├── configs/default.yaml     # single source of truth for all parameters
├── src/
│   ├── audio/                # loading, resampling, VAD
│   ├── features/              # log-Mel spectrogram extraction
│   ├── models/                 # mamba.py, gru.py, classifier.py
│   ├── training/                # training loop
│   ├── evaluation/               # metrics, confusion matrix, benchmarking
│   └── realtime/                  # microphone streaming + live demo
├── scripts/                  # CLI entry points (one per pipeline stage)
├── experiments/               # configs/results/logs/figures per run
├── tests/                     # pytest suite
└── paper/                     # algorithm_audit.md, methodology_notes.md,
                                # reproducibility.md, experimental_results.md
```

## Important notes on the Mamba implementation

The official CUDA-optimized `mamba-ssm` package (Gu & Dao's reference
implementation) requires an NVIDIA GPU and cannot install or run on
Apple Silicon. This project implements the Selective-SSM recurrence
directly in PyTorch, verified for both CPU and MPS execution. It is
mathematically consistent with the Selective SSM formulation but is
**not** a drop-in replacement for every kernel-level optimization in the
original CUDA implementation. See `paper/methodology_notes.md` for the
full technical distinction.

## Authors

Dev Choudhary (046), Abheer Bhati (177) — Symbiosis Institute of Computer
Studies and Research.
