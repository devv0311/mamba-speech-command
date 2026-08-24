# Methodology Notes

Implementation details recorded as they are built, for later use in the
IEEE paper's Methods section. This document contains implementation
facts, not experimental results — measured results (accuracy, latency,
memory, noise robustness, etc.) live in `paper/experimental_results.md`,
which as of commit `26bc1639d027a3cc697d3ebf2cb4eae27d233e99` reports
real measured values for every metric in the project brief, none marked
`NOT YET MEASURED`.

## 1. Task framing

Closed-vocabulary speech-command classification (not general ASR): given a
~1-second audio clip, predict one of 8 commands (yes, no, up, down, left,
right, go, stop). Framed as multi-class classification over a fixed
vocabulary, using the Google Speech Commands dataset's existing per-word
recordings — no transcription/decoding is performed.

## 2. Dataset

- **Source:** Google Speech Commands v0.02 (`speech_commands_v0.02.tar.gz`),
  official TensorFlow download URL:
  `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`.
- **Integrity verification:** SHA256 checksum verified against the value
  used by PyTorch's own `torchaudio.datasets.SpeechCommands` loader
  (`af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58`),
  copied verbatim from the torchaudio source rather than independently
  recomputed against a separate authority.
- **Subset used:** 8 commands (yes, no, up, down, left, right, go, stop),
  extracted from the full ~35-word dataset.
- **Split methodology:** official `validation_list.txt` / `testing_list.txt`
  files shipped inside the archive are used verbatim to assign utterances
  to val/test splits. Google constructs these lists on a per-speaker basis
  specifically to avoid speaker leakage across splits. Every utterance not
  listed in either file is assigned to train. A programmatic leakage check
  (no `speaker_id` appears in more than one split) is run and logged by
  `scripts/prepare_dataset.py` on every run.
- **Actual sample counts (measured, from a real run on 2026-08-24):**
  train 24,713 / val 2,967 / test 3,276 (30,956 total across the 8-class
  subset), roughly balanced per class (~2,900–3,300 train samples/class).
  Speaker-leakage check: PASS.

## 3. Audio preprocessing

- Mono, resampled to 16 kHz (native rate of the dataset; standard for
  speech-command recognition).
- Peak-normalized to [-1, 1].
- Fixed to 1.0 second duration via zero-padding (short clips) or
  center-cropping (long clips) — nearly all Speech Commands utterances are
  already ~1s, so this affects a small fraction of samples.
- WAV I/O implemented via `soundfile` (libsndfile), not torchaudio's own
  `load`/`save`. Recent torchaudio releases moved their file I/O backend to
  require the optional `torchcodec` package (which wraps FFmpeg);
  `soundfile` avoids that extra dependency and is used purely for reading
  raw PCM samples. torchaudio is still used for its tensor-level DSP
  transforms (`Resample`, `MelSpectrogram`), which are unaffected by this.

## 4. Feature extraction

Log-Mel spectrogram via `torchaudio.transforms.MelSpectrogram` + `log(x + eps)`:

| Parameter | Value | Rationale |
|---|---|---|
| Sample rate | 16,000 Hz | Dataset native rate |
| FFT size | 512 | ~32ms window, standard for speech |
| Window length | 400 samples (25ms) | Classic ASR frame size |
| Hop length | 160 samples (10ms) | Classic ASR frame stride |
| Mel bins | 40 | Compact bank sufficient for closed-vocabulary command recognition (vs. 80 typically used for full ASR) |
| f_min / f_max | 20 Hz / 8000 Hz | Full range up to Nyquist at 16kHz |

Resulting feature shape for a 1.0s clip: (40 mel bins, 101 time frames).

## 5. Mamba (Selective SSM) implementation

**Reference:** Gu, A. & Dao, T., "Mamba: Linear-Time Sequence Modeling with
Selective State Spaces" (arXiv:2312.00752).

**What we verified from the primary source before implementing** (fetched
the actual arXiv PDF and a secondary technical walkthrough, not relying on
memory alone):
- Zero-order-hold discretization: Ā = exp(ΔA); the paper's exact B̄ formula
  is (ΔA)⁻¹(exp(ΔA) − I)·ΔB, though the common (and the reference
  implementation's default) simplification B̄ ≈ ΔB is used here — this
  first-order approximation, not the exact ZOH integral, is what we
  implemented, and this is stated explicitly rather than silently presented
  as the exact formula.
- Selection mechanism: Δ, B, C become input-dependent via linear
  projections of the input (`x_proj`, `dt_proj` in code), with softplus
  applied to Δ to keep the step size positive.
- Recurrence: h_t = Ā_t·h_{t-1} + B̄_t·x_t; y_t = C_t·h_t.
- Block structure: input expands via a linear projection into two
  parallel branches (d_inner each); one branch goes through a causal
  depthwise 1D convolution + SiLU activation before entering the SSM; the
  other branch is a SiLU-gated multiplicative gate applied to the SSM
  output; a final linear projection returns to d_model; the whole block is
  wrapped in a residual connection. A per-channel skip/passthrough term D
  is added to the SSM output.

**What is NOT claimed:**
- This is NOT the official `mamba-ssm` CUDA kernel (state-spaces/mamba on
  GitHub). That kernel requires CUDA/nvcc and cannot install or run on
  Apple Silicon, so it is not a dependency of this project.
- The sequential scan here is implemented as an explicit Python `for` loop
  over the time dimension (in `SelectiveSSM.forward`, `src/models/mamba.py`),
  not a fused/parallel hardware-aware scan. This is mathematically
  equivalent to the recurrence but has real, measured performance
  implications — see Section 7.
- Where the original paper's block-level design details were not fully
  specified in the primary source excerpt we retrieved (e.g. exact
  ordering nuances), the implementation follows the standard convention
  used by the reference implementation and widely-reproduced community
  explanations, and this is noted here rather than presented as
  paper-verbatim.

## 6. GRU baseline

Standard `nn.GRU`, unidirectional (for causal comparability with Mamba),
same input feature projection (log-Mel → linear → d_model) and
classification head as the Mamba model. Hidden size tuned
(`configs/default.yaml: model_gru.hidden_size`) so parameter count lands in
the same order of magnitude as the Mamba model for a fair comparison.

**Measured parameter counts** (architecture-only figures, computed directly
from the instantiated models with the current `configs/default.yaml`; not
hardware-dependent): Mamba 140,264 params; GRU 113,832 params (ratio ≈
1.23×).

## 7. Known performance characteristic: sequential-scan overhead

**Preliminary sanity check** (measured on the development cloud sandbox,
2 CPU threads, no MPS available), NOT a reported research benchmark: a
single forward+backward pass through the full 4-layer Mamba model at
batch=64, sequence length=101 took approximately 19–21 seconds on CPU,
versus approximately 0.05s for the GRU baseline at batch=4 (Mamba was
roughly 7–8× slower than GRU in this informal check). Isolating the SSM
module alone, forward-only was fast (~0.14s for one 4-layer stack at
batch=64); the dominant cost is backpropagation through the ~101 sequential
Python-level loop iterations per layer, which is expected for a
non-fused/non-checkpointed sequential-scan implementation and is the
specific problem the reference CUDA kernel's hardware-aware parallel scan
algorithm is designed to solve.

**Real Apple Silicon (MPS) timing has since been measured** on the target
Mac and is reported in full in `paper/experimental_results.md` ("Full
benchmark sweep — latency" and "Full benchmark sweep — memory" sections,
commit `26bc1639d027a3cc697d3ebf2cb4eae27d233e99`): the same
sequential-scan overhead is visible there too — e.g. Mamba's full 8-epoch
training run took 77.9 min on MPS vs. 4.7 min for GRU under an identical
protocol (~16.5x slower), and Mamba's MPS driver-allocated memory is up
to 16.7x higher than GRU's at batch=64 — both consistent with, and now
confirmed beyond, this preliminary CPU-only sanity check. This section is
retained as the original preliminary finding that motivated timing-probing
the real hardware before committing to full training runs (see
`scripts/benchmark.py --probe-only` in `paper/reproducibility.md` step 4);
it is not itself the authoritative timing source once real MPS
measurements exist.

## 8. Real-time (microphone) preprocessing — capture alignment fix

The first live-microphone demo run (2026-08-24, `mamba_run1`, see
`paper/experimental_results.md` Experiment 4) produced valid, real
end-to-end latency measurements (all 10/10 predictions within the 200ms
criterion) but showed noticeably lower/more mixed prediction confidence
than the 94.5% clean-test-set accuracy, plus at least one label that did
not obviously match a straightforward read-through of the vocabulary.

**Root cause identified by direct comparison of the training-data path
(`SpeechCommandDataset.__getitem__`, `src/training/dataset.py`) against
the original live-inference path (`RealtimeCommandRecognizer.predict`,
`src/realtime/inference.py`):**

- Google Speech Commands clips are individually trimmed by the dataset's
  own recording protocol so each spoken word sits roughly centered in its
  fixed 1-second file; `fix_length()` (`src/audio/io.py`) only pads/trims
  by a small margin around that already-centered word.
- The original `MicrophoneListener` returned its rolling 1-second buffer
  the instant energy crossed `vad_energy_threshold` — i.e., at the exact
  onset of speech. This captures the word starting near the END of the
  returned window, not the middle. This was directly visible in the first
  run's waveform plot: the energy spike sat at approximately sample
  15,800 of 16,000 (the trailing ~12ms), not centered as training clips
  are.
- This is a real, verified framing-distribution mismatch between train
  and live inference, not a hypothesis — it follows directly from reading
  both code paths and the first run's own waveform plot.

**Fix applied (2026-08-24, after the first live run):**
`MicrophoneListener` gained a `post_trigger_wait_sec` parameter
(`configs/default.yaml: realtime.post_trigger_wait_sec`, default 0.35s)
and a `wait_for_utterance()` method: instead of snapshotting the instant
VAD fires, capture now waits an additional real wall-clock interval after
trigger before taking the 1-second snapshot, so the spoken word is more
likely to land nearer the window's center. `scripts/realtime_demo.py`'s
main loop applies the equivalent wait (via `time.sleep`, not a blocking
call, since the plot's event loop must keep servicing redraws) before
snapshotting.

**What this fix does NOT claim:** it is a heuristic alignment
improvement, not a reproduction of Google's exact per-utterance trimming
protocol. It has been re-validated with a second live microphone run
(26 predictions, post-fix, see `paper/experimental_results.md`
Experiment 4's "Post-fix re-verification" subsection) — mean confidence
rose from 38.1% to 73.8%, consistent with the capture-alignment
hypothesis, though several low-confidence predictions remain and this is
reported honestly as an open observation, not a fully resolved fix.

## 9. Device strategy

`src/training/device.py` selects MPS if `torch.backends.mps.is_available()`,
else falls back to CPU, and prints which device was actually selected
(never silently claims GPU acceleration). No MPS-specific operation
failures were encountered at any point, including full training runs
(`mamba_run1`, `gru_run1`), the full latency/memory benchmark sweep, and
the live-microphone real-time demo, all of which ran on MPS on the
target Mac — see `paper/experimental_results.md` for the measured
figures from each of those runs.
