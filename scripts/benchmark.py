#!/usr/bin/env python3
"""STEP 10 — Benchmark: measure real forward/backward/inference timing and
memory for both backbones, on both CPU and MPS (if available).

This is also used as a lightweight PRE-TRAINING PROBE: run with
--probe-only to get a few real timed batches (no full epoch) before
committing to a long training run, so epoch/batch-size choices for STEP 6/8
are based on a real measurement from this specific machine, not a guess.

Memory measurement methodology (real, measured — not estimated):
  - Process peak RSS: `resource.getrusage(RUSAGE_SELF).ru_maxrss`, sampled
    immediately before and after each (backbone, device, batch_size) run.
    On macOS this is reported in bytes (unlike Linux, where it is KB); we
    detect the platform and normalize to MB accordingly. This is the whole
    Python process's peak resident set size — it includes the model,
    activations, PyTorch/MPS runtime overhead, and everything else alive
    in-process, not an isolated "model-only" figure. It is monotonically
    non-decreasing within a single process, so `delta_rss_mb` (peak RSS
    after the run minus peak RSS before it) is reported per row as the
    best available real approximation of that row's incremental memory
    cost; the very first row's own baseline includes Python/PyTorch
    import overhead.
  - MPS allocator memory (MPS device rows only): `torch.mps.
    current_allocated_memory()` and `torch.mps.driver_allocated_memory()`,
    read immediately after the timed inference pass, before the next
    model is constructed. These are real PyTorch/Metal allocator
    counters, not estimates.
  - No CUDA-style `torch.cuda.max_memory_allocated` exists for MPS in this
    PyTorch version, so no such figure is fabricated; only the two real
    counters above are recorded for MPS, and process RSS is recorded for
    every row regardless of device.

Usage:
    python scripts/benchmark.py --probe-only          # quick real-hardware timing check
    python scripts/benchmark.py                        # full benchmark (used for the paper)
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import PROJECT_ROOT, load_config  # noqa: E402
from models.classifier import build_model, count_parameters  # noqa: E402


def peak_rss_mb() -> float:
    """Process peak RSS so far, in MB. ru_maxrss is bytes on macOS, KB on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = (1024 * 1024) if platform.system() == "Darwin" else 1024
    return raw / divisor


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

                # Reset MPS allocator counters and force a GC pass before each row so
                # per-row deltas are not contaminated by the previous row's tensors.
                gc.collect()
                if device.type == "mps":
                    torch.mps.empty_cache()
                rss_before_mb = peak_rss_mb()

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

                    rss_after_mb = peak_rss_mb()
                    mps_current_mb = mps_driver_mb = None
                    if device.type == "mps":
                        mps_current_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                        mps_driver_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)

                    row = {
                        "backbone": backbone, "device": str(device), "batch_size": batch_size,
                        "n_params": n_params,
                        "fwd_bwd_mean_sec": fb_mean, "fwd_bwd_min_sec": fb_min, "fwd_bwd_max_sec": fb_max,
                        "inference_mean_sec": inf_mean, "inference_min_sec": inf_min, "inference_max_sec": inf_max,
                        "inference_per_sample_ms": per_sample_inf_ms,
                        "process_peak_rss_mb_before": rss_before_mb,
                        "process_peak_rss_mb_after": rss_after_mb,
                        "process_delta_rss_mb": rss_after_mb - rss_before_mb,
                        "mps_current_allocated_mb": mps_current_mb,
                        "mps_driver_allocated_mb": mps_driver_mb,
                    }
                    results.append(row)
                    mem_str = f"delta_rss={rss_after_mb - rss_before_mb:7.1f}MB"
                    if device.type == "mps":
                        mem_str += f" mps_alloc={mps_current_mb:7.1f}MB mps_driver={mps_driver_mb:7.1f}MB"
                    print(f"{backbone:6s} {str(device):5s} batch={batch_size:3d} params={n_params:7d} "
                          f"fwd+bwd={fb_mean*1000:8.1f}ms  inference={inf_mean*1000:7.2f}ms "
                          f"({per_sample_inf_ms:.2f}ms/sample) {mem_str}")
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
