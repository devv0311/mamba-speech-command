#!/usr/bin/env python3
"""Quick standalone microphone smoke test — run this BEFORE realtime_demo.py
to confirm mic permission is granted and sounddevice can open a stream,
without needing a trained model loaded.

Usage:
    python scripts/mic_check.py
"""
from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    try:
        import sounddevice as sd
    except OSError as e:
        print(f"ERROR: sounddevice/PortAudio not available: {e}")
        return 1

    print("Available audio input devices:")
    print(sd.query_devices())
    print()

    print("Recording 2 seconds from the default input device...")
    print("(macOS will prompt for microphone permission the first time — allow it for Terminal / your IDE.)")

    try:
        recording = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: recording failed: {e}")
        print("Check System Settings > Privacy & Security > Microphone, and grant access to your terminal app / IDE.")
        return 1

    energy = float(np.mean(recording.astype(np.float64) ** 2))
    peak = float(np.abs(recording).max())
    print(f"Recorded 2.0s. Mean energy: {energy:.6f}, peak amplitude: {peak:.4f}")

    if peak < 1e-4:
        print("WARNING: recorded audio appears to be silence. Check your microphone input device/volume.")
        return 1

    print("PASS — microphone capture is working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
