"""Real-time microphone capture + inference pipeline.

Captures audio in a rolling buffer via sounddevice, uses a simple
energy-based voice activity detector to decide when a full utterance has
likely been spoken, then runs the same preprocessing -> feature extraction
-> model pipeline used everywhere else in this project (no separate/
duplicated inference path) and measures each stage's real wall-clock time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
import torch

from audio.io import fix_length, peak_normalize
from features.logmel import LogMelExtractor


@dataclass
class InferenceResult:
    predicted_command: str
    probabilities: dict  # command -> probability
    preprocessing_time_sec: float
    inference_time_sec: float
    total_latency_sec: float
    waveform: np.ndarray
    log_mel: np.ndarray
    state_trace: object | None = None  # populated only for Mamba, when requested


@dataclass
class MicrophoneListener:
    """Continuously records audio into a rolling ring buffer and exposes a
    simple energy-based "utterance detected" trigger.

    IMPORTANT — capture alignment (fixed 2026-08-24 after the first live
    demo run showed low/mixed confidence vs. the 94.5% clean test-set
    accuracy): the training data (Google Speech Commands) has each spoken
    word roughly CENTERED inside its fixed 1-second clip, because the
    dataset's own recording protocol trims silence around each utterance.
    If we naively return the rolling buffer's current 1-second window the
    instant VAD energy crosses threshold, the word's onset is caught right
    at the trailing edge of the window (verified in the first demo run's
    waveform plot: the energy spike sat at ~15800/16000 samples), which is
    a framing distribution the model never saw in training. To better
    match training-time framing, `wait_for_utterance()` triggers on rising
    energy but then WAITS an additional ~0.5s (pre/post capture split
    below) before taking the snapshot, so the spoken word lands closer to
    the middle of the returned window instead of at its tail. This is a
    heuristic alignment fix, not a guarantee of centering — documented
    honestly as such, not claimed to reproduce the dataset's exact
    trimming protocol.
    """
    sample_rate: int
    buffer_duration_sec: float
    vad_energy_threshold: float
    device: int | None = None
    post_trigger_wait_sec: float = 0.35  # how long to keep recording after VAD fires, before snapshotting

    _buffer: np.ndarray = field(init=False, repr=False)
    _stream: sd.InputStream | None = field(default=None, init=False, repr=False)
    _write_pos: int = field(default=0, init=False, repr=False)
    _latest_energy: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        n_samples = int(self.sample_rate * self.buffer_duration_sec)
        self._buffer = np.zeros(n_samples, dtype=np.float32)

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            # Non-fatal stream status (e.g. buffer overrun) — recorded, not silently ignored.
            print(f"[microphone] stream status: {status}")
        mono = indata[:, 0] if indata.ndim > 1 else indata
        n = len(mono)
        if n >= len(self._buffer):
            self._buffer[:] = mono[-len(self._buffer):]
        else:
            self._buffer = np.roll(self._buffer, -n)
            self._buffer[-n:] = mono
        self._latest_energy = float(np.mean(mono.astype(np.float64) ** 2))

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            device=self.device, callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def is_speech_detected(self) -> bool:
        return self._latest_energy >= self.vad_energy_threshold

    def snapshot(self) -> np.ndarray:
        return self._buffer.copy()

    def wait_for_utterance(self, poll_interval_sec: float = 0.01) -> np.ndarray:
        """Block until VAD fires, then keep recording for
        `post_trigger_wait_sec` more before snapshotting, so the spoken
        word is better centered in the returned window (see class
        docstring). Real wall-clock wait via time.sleep — not simulated.
        """
        while not self.is_speech_detected():
            time.sleep(poll_interval_sec)
        time.sleep(self.post_trigger_wait_sec)
        return self.snapshot()


class RealtimeCommandRecognizer:
    """Wraps a trained model + the standard preprocessing pipeline for
    single-utterance real-time prediction, with real per-stage timing."""

    def __init__(self, model, cfg: dict, device: torch.device):
        self.model = model.to(device).eval()
        self.device = device
        self.cfg = cfg
        self.sample_rate = cfg["audio"]["sample_rate"]
        self.target_len = int(cfg["dataset"]["target_duration_sec"] * self.sample_rate)
        self.extractor = LogMelExtractor(cfg)
        self.class_names = sorted(cfg["dataset"]["commands"])

    @torch.no_grad()
    def predict(self, waveform_np: np.ndarray, capture_state_trace: bool = False) -> InferenceResult:
        t_start = time.perf_counter()

        waveform = torch.from_numpy(waveform_np.astype(np.float32))
        waveform = peak_normalize(waveform)
        waveform = fix_length(waveform, self.target_len)
        log_mel = self.extractor(waveform)

        t_preproc_done = time.perf_counter()

        log_mel_batched = log_mel.unsqueeze(0).to(self.device)  # (1, n_mels, n_frames)

        if capture_state_trace and getattr(self.model.encoder, "supports_state_trace", False):
            logits, trace = self.model(log_mel_batched, return_state_trace=True)
        else:
            logits = self.model(log_mel_batched)
            trace = None

        if self.device.type == "mps":
            torch.mps.synchronize()
        t_inference_done = time.perf_counter()

        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_idx = int(probs.argmax())

        return InferenceResult(
            predicted_command=self.class_names[pred_idx],
            probabilities={name: float(probs[i]) for i, name in enumerate(self.class_names)},
            preprocessing_time_sec=t_preproc_done - t_start,
            inference_time_sec=t_inference_done - t_preproc_done,
            total_latency_sec=t_inference_done - t_start,
            waveform=waveform.numpy(),
            log_mel=log_mel.numpy(),
            state_trace=trace,
        )
