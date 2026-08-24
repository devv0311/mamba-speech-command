#!/usr/bin/env python3
"""STEP 11 — Real-time microphone speech-command demo.

Launches a live matplotlib window showing:
  1. Microphone status (LISTENING / PROCESSING / PREDICTED)
  2. Current audio waveform
  3. Extracted log-Mel spectrogram
  4. (Mamba only) a visualization of the SSM hidden-state trajectory,
     generated from an ACTUAL forward pass — not faked.
  5. Predicted command + per-class probabilities (real softmax output)
  6. Real measured timing: preprocessing / inference / total latency

Trigger: simple energy-based voice activity detection — when the rolling
microphone buffer's energy exceeds configs/default.yaml:
realtime.vad_energy_threshold, the current 1-second buffer is treated as
one utterance and run through the full pipeline.

Usage:
    python scripts/realtime_demo.py --run-name mamba_run1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
try:
    matplotlib.use("MacOSX")  # native interactive backend on macOS
except Exception:  # noqa: BLE001
    pass  # fall back to matplotlib's own default interactive backend for this platform
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import PROJECT_ROOT, load_config  # noqa: E402
from models.classifier import build_model  # noqa: E402
from realtime.inference import MicrophoneListener, RealtimeCommandRecognizer  # noqa: E402
from training.device import select_device  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--refractory-sec", type=float, default=1.5,
                         help="Minimum time between consecutive predictions, to avoid re-triggering on the same utterance")
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "experiments" / "results" / args.run_name
    with open(run_dir / "summary.json") as f:
        run_summary = json.load(f)
    backbone = run_summary["backbone"]
    checkpoint_path = Path(run_summary["checkpoint_path"])

    cfg = load_config(args.config)
    device = select_device(cfg["device"]["prefer"], cfg["device"]["fallback"])

    model = build_model(cfg, backbone=backbone)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded {backbone} checkpoint from {checkpoint_path} onto {device}")

    recognizer = RealtimeCommandRecognizer(model, cfg, device)
    listener = MicrophoneListener(
        sample_rate=cfg["audio"]["sample_rate"],
        buffer_duration_sec=cfg["dataset"]["target_duration_sec"],
        vad_energy_threshold=cfg["realtime"]["vad_energy_threshold"],
        device=cfg["realtime"]["input_device"],
        post_trigger_wait_sec=cfg["realtime"].get("post_trigger_wait_sec", 0.35),
    )

    try:
        listener.start()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not open microphone stream: {e}")
        print("Check System Settings > Privacy & Security > Microphone permissions for Terminal/your IDE.")
        return 1

    class_names = recognizer.class_names
    latency_target_ms = cfg["realtime"]["latency_target_ms"]

    # --- Figure layout ---
    fig, axes = plt.subplot_mosaic(
        [["status", "status"], ["wave", "mel"], ["state", "state"], ["bars", "bars"]],
        figsize=(10, 11), height_ratios=[0.6, 2, 1.6, 2.2],
    )
    fig.suptitle(f"Mamba Speech Command Demo — backbone: {backbone}", fontsize=13)

    status_text = axes["status"].text(0.02, 0.5, "LISTENING", fontsize=20, va="center", color="tab:blue")
    timing_text = axes["status"].text(0.55, 0.5, "", fontsize=10, va="center", family="monospace")
    axes["status"].axis("off")

    (wave_line,) = axes["wave"].plot(np.zeros(listener._buffer.shape[0]))
    axes["wave"].set_ylim(-1.05, 1.05)
    axes["wave"].set_title("Waveform")

    mel_im = axes["mel"].imshow(np.zeros((cfg["features"]["n_mels"], 10)), aspect="auto", origin="lower", cmap="magma")
    axes["mel"].set_title("Log-Mel spectrogram")

    (state_line,) = axes["state"].plot([], [])
    axes["state"].set_title("Mamba SSM hidden-state trajectory (layer 1, channel 0, real forward pass)"
                             if backbone == "mamba" else "State trajectory visualization not available for GRU backbone")
    axes["state"].set_xlabel("Time frame")
    axes["state"].set_ylabel("Hidden state value")

    bar_container = axes["bars"].bar(class_names, [0] * len(class_names), color="tab:gray")
    axes["bars"].set_ylim(0, 1.0)
    axes["bars"].set_title("Predicted command probabilities (real softmax output)")
    axes["bars"].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    plt.ion()
    plt.show()

    print("Listening. Speak one of:", ", ".join(class_names))
    print(f"Latency criterion: total end-to-end latency <= {latency_target_ms} ms counts as real-time.")
    print("Close the plot window or press Ctrl+C in the terminal to stop.")

    last_prediction_time = 0.0

    try:
        while plt.fignum_exists(fig.number):
            if listener.is_speech_detected() and (time.time() - last_prediction_time) > args.refractory_sec:
                status_text.set_text("PROCESSING")
                status_text.set_color("tab:orange")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

                # Wait out the post-trigger window (real wall-clock delay) so the
                # spoken word is better centered in the captured buffer instead of
                # sitting at the trailing edge — see MicrophoneListener docstring.
                time.sleep(listener.post_trigger_wait_sec)
                buf = listener.snapshot()
                capture_trace = backbone == "mamba"
                result = recognizer.predict(buf, capture_state_trace=capture_trace)
                last_prediction_time = time.time()

                # Update waveform
                wave_line.set_ydata(result.waveform)

                # Update spectrogram
                mel_im.set_data(result.log_mel)
                mel_im.set_extent([0, result.log_mel.shape[1], 0, result.log_mel.shape[0]])
                mel_im.set_clim(result.log_mel.min(), result.log_mel.max())

                # Update state trace (Mamba only): layer 1, channel 0, across time
                if result.state_trace is not None:
                    layer0_trace = result.state_trace[0]["hidden_state_trajectory"]  # (1, L, d_inner, d_state)
                    channel0 = layer0_trace[0, :, 0, 0].cpu().numpy()  # first inner channel, first state dim
                    state_line.set_data(np.arange(len(channel0)), channel0)
                    axes["state"].relim()
                    axes["state"].autoscale_view()

                # Update prediction bars
                probs = [result.probabilities[c] for c in class_names]
                for bar, p in zip(bar_container, probs):
                    bar.set_height(p)
                    bar.set_color("tab:green" if p == max(probs) else "tab:gray")

                total_ms = result.total_latency_sec * 1000
                realtime_ok = total_ms <= latency_target_ms
                status_text.set_text(f"PREDICTED: {result.predicted_command.upper()} — {max(probs)*100:.1f}%")
                status_text.set_color("tab:green" if realtime_ok else "tab:red")
                timing_text.set_text(
                    f"preprocess={result.preprocessing_time_sec*1000:6.1f}ms  "
                    f"inference={result.inference_time_sec*1000:6.1f}ms  "
                    f"total={total_ms:6.1f}ms  "
                    f"({'within' if realtime_ok else 'EXCEEDS'} {latency_target_ms}ms target)"
                )

                print(f"{result.predicted_command:8s} {max(probs)*100:5.1f}%  "
                      f"preprocess={result.preprocessing_time_sec*1000:.1f}ms "
                      f"inference={result.inference_time_sec*1000:.1f}ms "
                      f"total={total_ms:.1f}ms")

            else:
                status_text.set_text("LISTENING")
                status_text.set_color("tab:blue")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.05)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        listener.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
