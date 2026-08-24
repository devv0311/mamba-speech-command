#!/usr/bin/env python3
"""Download the Google Speech Commands dataset (v0.02).

Reproducible dataset acquisition: downloads the official archive, verifies
its checksum against the official published checksum, and extracts it.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import PROJECT_ROOT, load_config  # noqa: E402

# SHA256 for speech_commands_v0.02.tar.gz, copied verbatim from PyTorch's own
# torchaudio dataset loader (_CHECKSUMS dict), which uses it to verify this
# exact download URL:
#   https://github.com/pytorch/audio/blob/main/src/torchaudio/datasets/speechcommands.py
# This is a primary-source-verified value, not independently computed by us
# (we did not download and hash the ~2.4GB archive ourselves to cross-check it).
OFFICIAL_SHA256 = "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58"


def sha256sum(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_progress(url: str, dest: Path) -> None:
    def _report(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        pct = min(100.0, downloaded / total_size * 100) if total_size > 0 else 0.0
        sys.stdout.write(f"\r  {dest.name}: {pct:5.1f}% ({downloaded/1e6:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_report)
    sys.stdout.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-checksum", action="store_true",
                         help="Skip MD5 verification (not recommended)")
    parser.add_argument("--force", action="store_true", help="Re-download even if archive exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = PROJECT_ROOT / cfg["dataset"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    url = cfg["dataset"]["source_url"]
    archive_path = raw_dir / "speech_commands_v0.02.tar.gz"
    extract_dir = raw_dir / "speech_commands_v0.02"

    if extract_dir.exists() and any(extract_dir.iterdir()) and not args.force:
        print(f"Dataset already extracted at {extract_dir}. Use --force to re-download.")
        return 0

    if not archive_path.exists() or args.force:
        print(f"Downloading {url}")
        print(f"  -> {archive_path}")
        download_with_progress(url, archive_path)
    else:
        print(f"Archive already present at {archive_path}, skipping download.")

    if not args.skip_checksum:
        print("Verifying SHA256 checksum...")
        actual = sha256sum(archive_path)
        if actual != OFFICIAL_SHA256:
            print(f"CHECKSUM MISMATCH: expected {OFFICIAL_SHA256}, got {actual}")
            print("The archive may be corrupted or the upstream file changed. Aborting extraction.")
            return 1
        print(f"  SHA256 OK: {actual}")
    else:
        print("Skipping checksum verification (--skip-checksum).")

    print(f"Extracting to {extract_dir} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    n_files = sum(1 for _ in extract_dir.rglob("*.wav"))
    print(f"Done. {n_files} .wav files extracted under {extract_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
