#!/usr/bin/env python3
"""STEP 10 — Benchmark: measure real forward/backward/inference timing and
memory for both backbones, on both CPU and MPS (if available).

This is also used as a lightweight PRE-TRAINING PROBE: run with
--probe-only to get a few real timed batches (no full epoch) before
committing to a long training run, so epoch/batch-size choices for STEP 6/8
are based on a real measurement from this specific machine, not a guess.

Usage:
    python scripts/benchmark.py --probe-only          # quick real-hardware timing check
    python scripts/benchmark.py                        # full benchmark (used for the paper)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import PROJECT_ROOT, load_config  # noqa: E402
from models.classifier import build_model, count_parameters  # noqa: E402


def time_forward_backward(model, x, device, n_warmup=2, n_measure=5):
    model = model.to(device)
    x = x.to(device)

    for _ in range(n_warmup):
        out = model(x)
        out.sum().backward()
        model.zero_grad()
        if device.type == "mps":
            torch.mps.synchronize()

    times = []
    for _ in range(n_measure):
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        out = model(x)
        out.sum().backward()
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
        model.zero_grad()

    return sum(times) / len(times), min(times), max(times)


def time_inference(model, x, device, n_warmup=3, n_measure=10):
    model = model.to(device).eval()
    x = x.to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
            if device.type == "mps":
                torch.mps.synchronize()

        times = []
        for _ in range(n_measure):
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)

    return sum(times) / len(times), min(times), max(times)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--probe-only", action="store_true",
                         help="Quick timing check with a few batches only — no full benchmark sweep.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_mels = cfg["features"]["n_mels"]
    target_duration = cfg["dataset"]["target_duration_sec"]
    sample_rate = cfg["audio"]["sample_rate"]
    hop_length = cfg["features"]["hop_length"]
    n_frames = int(target_duration * sample_rate) // hop_length + 1

    devices = []
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    devices.append(torch.device("cpu"))

    batch_sizes = args.batch_sizes or ([8] if args.probe_only else [1, 8, 16, cfg["training"]["batch_size"]])

    results = []
    for backbone in ["mamba", "gru"]:
        for device in devices:
            for batch_size in batch_sizes:
                torch.manual_seed(cfg["project"]["seed"])
                model = build_model(cfg, backbone=backbone)
                n_params = count_parameters(model)
                x = torch.randn(batch_size, n_mels, n_frames)

                try:
                    fb_mean, fb_min, fb_max = time_forward_backward(
                        model, x, device, n_warmup=1 if args.probe_only else 2,
                        n_measure=2 if args.probe_only else 5,
                    )
                    inf_mean, inf_min, inf_max = time_inference(
                        model, x, device, n_warmup=2 if args.probe_only else 3,
                        n_measure=3 if args.probe_only else 10,
                    )
                    per_sample_inf_ms = (inf_mean / batch_size) * 1000

                    row = {
                        "backbone": backbone, "device": str(device), "batch_size": batch_size,
                        "n_params": n_params,
                        "fwd_bwd_mean_sec": fb_mean, "fwd_bwd_min_sec": fb_min, "fwd_bwd_max_sec": fb_max,
                        "inference_mean_sec": inf_mean, "inference_min_sec": inf_min, "inference_max_sec": inf_max,
                        "inference_per_sample_ms": per_sample_inf_ms,
                    }
                    results.append(row)
                    print(f"{backbone:6s} {str(device):5s} batch={batch_size:3d} params={n_params:7d} "
                          f"fwd+bwd={fb_mean*1000:8.1f}ms  inference={inf_mean*1000:7.2f}ms "
                          f"({per_sample_inf_ms:.2f}ms/sample)")
                except Exception as e:  # noqa: BLE001
                    print(f"{backbone:6s} {str(device):5s} batch={batch_size:3d}: FAILED ({e})")
                    results.append({"backbone": backbone, "device": str(device), "batch_size": batch_size, "error": str(e)})

                if args.probe_only:
                    # Estimate real full-epoch time using THIS machine's measured fwd+bwd time.
                    train_manifest = PROJECT_ROOT / cfg["dataset"]["processed_dir"] / "train.csv"
                    if train_manifest.exists():
                        import csv  # noqa: PLC0415
                        with open(train_manifest) as f:
                            n_train = sum(1 for _ in csv.reader(f)) - 1
                        n_batches = max(1, n_train // batch_size)
                        est_epoch_sec = fb_mean * n_batches
                        print(f"    -> estimated epoch time on {device} at batch={batch_size}: "
                              f"{est_epoch_sec:.1f}s ({est_epoch_sec/60:.1f} min) for {n_train} train samples")

    out_path = PROJECT_ROOT / "experiments" / "results" / ("benchmark_probe.json" if args.probe_only else "benchmark_full.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
