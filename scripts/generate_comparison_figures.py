#!/usr/bin/env python3
"""Generate the Mamba-vs-GRU comparison figures (accuracy, latency, and, if
available, noise robustness) from real saved run artifacts — never from
invented numbers. Requires both runs' summary.json/test_metrics.json (and
noise_robustness.json / benchmark_full.json where applicable) to already
exist on disk.

Usage:
    python scripts/generate_comparison_figures.py --mamba-run mamba_run1 --gru-run gru_run1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import PROJECT_ROOT  # noqa: E402
from evaluation.plots import plot_model_comparison_bar, plot_noise_robustness  # noqa: E402


def load_run(run_name: str) -> tuple[dict, dict]:
    run_dir = PROJECT_ROOT / "experiments" / "results" / run_name
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)
    with open(run_dir / "test_metrics.json") as f:
        metrics = json.load(f)
    return summary, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mamba-run", type=str, required=True)
    parser.add_argument("--gru-run", type=str, required=True)
    args = parser.parse_args()

    mamba_summary, mamba_metrics = load_run(args.mamba_run)
    gru_summary, gru_metrics = load_run(args.gru_run)

    fig_dir = PROJECT_ROOT / "experiments" / "figures"

    # Accuracy / F1 comparison
    plot_model_comparison_bar(
        labels=["Accuracy", "Macro F1"],
        values_by_model={
            "Mamba": [mamba_metrics["accuracy"], mamba_metrics["macro_f1"]],
            "GRU": [gru_metrics["accuracy"], gru_metrics["macro_f1"]],
        },
        ylabel="Score", title="Mamba vs GRU — Test Set Performance",
        out_path=fig_dir / "mamba_vs_gru_accuracy.png",
    )
    print(f"Wrote {fig_dir / 'mamba_vs_gru_accuracy.png'}")

    # Training time comparison
    plot_model_comparison_bar(
        labels=["Training time (min)"],
        values_by_model={
            "Mamba": [mamba_summary["total_training_time_sec"] / 60],
            "GRU": [gru_summary["total_training_time_sec"] / 60],
        },
        ylabel="Minutes", title=f"Mamba vs GRU — Training Time ({mamba_summary['epochs_run']} epochs, {mamba_summary['device']})",
        out_path=fig_dir / "mamba_vs_gru_training_time.png",
    )
    print(f"Wrote {fig_dir / 'mamba_vs_gru_training_time.png'}")

    # Parameter count comparison
    plot_model_comparison_bar(
        labels=["Parameters"],
        values_by_model={"Mamba": [mamba_summary["n_params"]], "GRU": [gru_summary["n_params"]]},
        ylabel="Count", title="Mamba vs GRU — Parameter Count",
        out_path=fig_dir / "mamba_vs_gru_params.png",
    )
    print(f"Wrote {fig_dir / 'mamba_vs_gru_params.png'}")

    # Noise robustness, if both runs have it
    mamba_noise_path = PROJECT_ROOT / "experiments" / "results" / args.mamba_run / "noise_robustness.json"
    gru_noise_path = PROJECT_ROOT / "experiments" / "results" / args.gru_run / "noise_robustness.json"
    if mamba_noise_path.exists() and gru_noise_path.exists():
        with open(mamba_noise_path) as f:
            mamba_noise = json.load(f)["results"]
        with open(gru_noise_path) as f:
            gru_noise = json.load(f)["results"]

        conditions = list(mamba_noise.keys())
        plot_noise_robustness(
            conditions=conditions,
            accuracy_by_model={
                "Mamba": [mamba_noise[c]["accuracy"] for c in conditions],
                "GRU": [gru_noise[c]["accuracy"] for c in conditions],
            },
            out_path=fig_dir / "noise_robustness_comparison.png",
        )
        print(f"Wrote {fig_dir / 'noise_robustness_comparison.png'}")
    else:
        print("Noise robustness results not found for one or both runs — run scripts/noise_experiment.py first. Skipping that figure.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
