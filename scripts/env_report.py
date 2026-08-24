#!/usr/bin/env python3
"""Environment report — run this FIRST, before anything else.

Prints Python/PyTorch/macOS/hardware/MPS details in the exact format required
by the project's reproducibility rules, and runs one real MPS tensor op (not
just an availability check) so the report reflects actual measured behavior,
not an assumption.

Usage:
    python scripts/env_report.py
    python scripts/env_report.py --json experiments/logs/env_report.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone


def _safe(fn, default="UNKNOWN"):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - deliberately broad for a diagnostic script
        return f"{default} (error: {e})"


def get_macos_version() -> str:
    try:
        out = subprocess.run(["sw_vers"], capture_output=True, text=True, check=True)
        return out.stdout.strip().replace("\n", " | ")
    except Exception as e:  # noqa: BLE001
        return f"UNKNOWN (sw_vers failed: {e})"


def get_ram_gb() -> str:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
        bytes_ram = int(out.stdout.strip())
        return f"{bytes_ram / (1024**3):.1f} GB"
    except Exception as e:  # noqa: BLE001
        return f"UNKNOWN (sysctl failed: {e})"


def get_chip() -> str:
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"UNKNOWN (sysctl failed: {e})"


def get_git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "NOT_A_GIT_REPO_OR_GIT_UNAVAILABLE"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None, help="Optional path to also write JSON output")
    args = parser.parse_args()

    report: dict[str, str] = {}
    report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    report["python_version"] = sys.version.replace("\n", " ")
    report["python_executable"] = sys.executable
    report["platform"] = platform.platform()
    report["machine"] = platform.machine()
    report["macos_version"] = get_macos_version()
    report["chip"] = get_chip()
    report["ram"] = get_ram_gb()
    report["git_commit"] = get_git_commit()

    try:
        import torch  # noqa: PLC0415
        report["torch_version"] = torch.__version__
        report["torch_installed"] = "True"

        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        report["mps_available"] = str(mps_available)
        report["mps_built"] = str(mps_built)

        if mps_available:
            try:
                x = torch.rand(1024, 1024, device="mps")
                y = torch.rand(1024, 1024, device="mps")
                z = (x @ y).sum().item()
                report["mps_tensor_op_test"] = f"PASS (1024x1024 matmul+sum on MPS -> scalar {z:.4f})"
                report["selected_device"] = "mps"
            except Exception as e:  # noqa: BLE001
                report["mps_tensor_op_test"] = f"FAIL ({e})"
                report["selected_device"] = "cpu (MPS op failed, fell back)"
        else:
            report["mps_tensor_op_test"] = "SKIPPED (MPS not available)"
            report["selected_device"] = "cpu"

        # Also verify plain CPU tensor ops work regardless.
        try:
            xc = torch.rand(1024, 1024)
            yc = torch.rand(1024, 1024)
            _ = (xc @ yc).sum().item()
            report["cpu_tensor_op_test"] = "PASS"
        except Exception as e:  # noqa: BLE001
            report["cpu_tensor_op_test"] = f"FAIL ({e})"

    except ImportError:
        report["torch_installed"] = "False"
        report["torch_version"] = "NOT_INSTALLED"
        report["mps_available"] = "N/A (torch not installed)"
        report["mps_built"] = "N/A (torch not installed)"
        report["mps_tensor_op_test"] = "N/A (torch not installed)"
        report["selected_device"] = "N/A (torch not installed)"

    for key in ("torchaudio", "numpy", "sounddevice", "yaml"):
        try:
            mod = __import__(key)
            report[f"{key}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            report[f"{key}_version"] = "NOT_INSTALLED"

    # ---- print in the exact format requested by the project instructions ----
    print("=" * 70)
    print("ENVIRONMENT REPORT")
    print("=" * 70)
    print(f"Timestamp (UTC): {report['timestamp_utc']}")
    print(f"Python:          {report['python_version']}")
    print(f"Python exe:      {report['python_executable']}")
    print(f"PyTorch:         {report['torch_version']}")
    print(f"macOS:           {report['macos_version']}")
    print(f"Machine:         {report['machine']} ({report['chip']})")
    print(f"RAM:             {report['ram']}")
    print(f"MPS available:   {report['mps_available']}")
    print(f"MPS built:       {report['mps_built']}")
    print(f"MPS tensor test: {report['mps_tensor_op_test']}")
    print(f"CPU tensor test: {report.get('cpu_tensor_op_test', 'N/A')}")
    print(f"Selected device: {report['selected_device']}")
    print(f"torchaudio:      {report.get('torchaudio_version')}")
    print(f"numpy:           {report.get('numpy_version')}")
    print(f"sounddevice:     {report.get('sounddevice_version')}")
    print(f"git commit:      {report['git_commit']}")
    print("=" * 70)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"JSON report written to: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
