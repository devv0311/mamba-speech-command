# Data directory

This directory is intentionally empty in version control (see `.gitignore`).

Run `python scripts/download_dataset.py` to fetch the Google Speech Commands
dataset (v0.02) into `data/raw/`, then `python scripts/prepare_dataset.py`
to produce the filtered 8-class, speaker-disjoint train/val/test manifests
in `data/processed/`.

Exact dataset provenance (source URL, version, sample counts per class,
split methodology) is documented in `paper/methodology_notes.md` once
`prepare_dataset.py` has actually been run — counts are not pre-filled here
to avoid stating unverified numbers.
