#!/usr/bin/env python3
"""Prepare the 8-command manifest with a speaker-disjoint train/val/test split.

Uses the OFFICIAL validation_list.txt and testing_list.txt files shipped
inside the Speech Commands archive to assign utterances to val/test (Google
constructs these lists so that no speaker appears in more than one split,
per the dataset's own documentation). Every utterance not listed in either
file is assigned to train.

Writes three manifest CSVs (train.csv, val.csv, test.csv) to
dataset.processed_dir, each with columns: filepath, label, speaker_id.

Usage:
    python scripts/prepare_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import PROJECT_ROOT, load_config  # noqa: E402


def read_split_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def speaker_id_from_filename(filename: str) -> str:
    # Google Speech Commands filenames are formatted <speaker_hash>_nohash_<n>.wav
    return filename.split("_nohash_")[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    commands = set(ds_cfg["commands"])

    raw_root = PROJECT_ROOT / ds_cfg["raw_dir"] / "speech_commands_v0.02"
    processed_dir = PROJECT_ROOT / ds_cfg["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        print(f"ERROR: raw dataset not found at {raw_root}. Run scripts/download_dataset.py first.")
        return 1

    val_list = read_split_list(raw_root / "validation_list.txt")
    test_list = read_split_list(raw_root / "testing_list.txt")

    if not val_list or not test_list:
        print("ERROR: validation_list.txt / testing_list.txt not found or empty inside the archive.")
        print("Cannot guarantee the official speaker-disjoint split without them. Aborting.")
        return 1

    rows = {"train": [], "val": [], "test": []}
    per_class_counts = {split: Counter() for split in rows}

    for command_dir in sorted(raw_root.iterdir()):
        if not command_dir.is_dir() or command_dir.name not in commands:
            continue
        label = command_dir.name
        for wav_path in sorted(command_dir.glob("*.wav")):
            rel_key = f"{label}/{wav_path.name}"   # matches the format used in the official list files
            speaker = speaker_id_from_filename(wav_path.name)

            if rel_key in val_list:
                split = "val"
            elif rel_key in test_list:
                split = "test"
            else:
                split = "train"

            rows[split].append({
                "filepath": str(wav_path.relative_to(PROJECT_ROOT)),
                "label": label,
                "speaker_id": speaker,
            })
            per_class_counts[split][label] += 1

    # Speaker-leakage check: no speaker_id should appear in more than one split.
    speakers_by_split = {
        split: {r["speaker_id"] for r in rows[split]} for split in rows
    }
    leakage = (
        (speakers_by_split["train"] & speakers_by_split["val"])
        | (speakers_by_split["train"] & speakers_by_split["test"])
        | (speakers_by_split["val"] & speakers_by_split["test"])
    )

    for split, records in rows.items():
        out_path = processed_dir / f"{split}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "label", "speaker_id"])
            writer.writeheader()
            writer.writerows(records)
        print(f"{split}: {len(records)} samples -> {out_path}")
        for label in sorted(commands):
            print(f"    {label}: {per_class_counts[split][label]}")

    print()
    if leakage:
        print(f"WARNING: speaker leakage detected across splits! {len(leakage)} overlapping speaker id(s).")
    else:
        print("Speaker-leakage check: PASS (no speaker appears in more than one split).")

    total = sum(len(r) for r in rows.values())
    print(f"\nTotal samples across all splits (8-command subset): {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
