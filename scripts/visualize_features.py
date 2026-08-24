#!/usr/bin/env python3
"""Generate sample waveform + log-Mel spectrogram figures for a few dataset examples.

Verifies feature extraction end-to-end on real audio and produces
paper-ready figures (part of STEP 3).

Usage:
    python scripts/visualize_features.py --n 4
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import PROJECT_ROOT, load_config  # noqa: E402
from audio.io import load_wav, peak_normalize, fix_length  # noqa: E402
from features.logmel import LogMelExtractor  # noqa: E402
from features.visualize import plot_waveform_and_spectrogram  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--n", type=int, default=4, help="Number of samples to visualize")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=str, default="experiments/figures/feature_samples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = PROJECT_ROOT / cfg["dataset"]["processed_dir"] / f"{args.split}.csv"

    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}. Run scripts/prepare_dataset.py first.")
        return 1

    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"ERROR: manifest {manifest_path} is empty.")
        return 1

    sample_rate = cfg["audio"]["sample_rate"]
    target_len = int(cfg["dataset"]["target_duration_sec"] * sample_rate)
    extractor = LogMelExtractor(cfg)

    # pick one example per distinct label, up to args.n
    seen_labels = set()
    chosen = []
    for row in rows:
        if row["label"] not in seen_labels:
            chosen.append(row)
            seen_labels.add(row["label"])
        if len(chosen) >= args.n:
            break

    out_dir = PROJECT_ROOT / args.out_dir
    for row in chosen:
        wav_path = PROJECT_ROOT / row["filepath"]
        waveform = load_wav(wav_path, target_sample_rate=sample_rate)
        waveform = peak_normalize(waveform)
        waveform = fix_length(waveform, target_len)

        log_mel = extractor(waveform)

        assert torch.isfinite(log_mel).all(), f"Non-finite values in log-Mel for {wav_path}"  # noqa: F821

        out_path = out_dir / f"{row['label']}_{wav_path.stem}.png"
        plot_waveform_and_spectrogram(
            waveform, log_mel, sample_rate, cfg["features"]["hop_length"],
            title=row["label"], out_path=out_path,
        )
        print(f"{row['label']:8s} shape={tuple(log_mel.shape)} finite=OK -> {out_path}")

    print(f"\n{len(chosen)} figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
