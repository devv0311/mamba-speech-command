#!/usr/bin/env python3
"""STEP 8 — GRU baseline training, using the identical protocol as train_mamba.py
(same dataset manifests, same batch size / optimizer / schedule / early stopping)
so the comparison against Mamba is fair.

Usage:
    python scripts/train_gru.py
    python scripts/train_gru.py --epochs 15 --batch-size 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import load_config  # noqa: E402
from training.run import run_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    try:
        run_training(cfg, backbone="gru", run_name=args.run_name)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
