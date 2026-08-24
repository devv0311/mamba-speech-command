# Algorithm Audit — Existing Literature vs. Mamba

**Purpose:** Verify, before implementation, that Mamba / Selective State Space
Models are not already represented in the project's existing 25-paper
literature set, and establish that Mamba is a distinct algorithmic
contribution rather than a redundant re-selection.

**Source file:** `Voice_Assistant_AI_ML_Papers_2022-2025.xlsx`
(1 sheet: "Voice Assistant AI-ML Papers", 25 papers, years 2022–2025,
columns: No., Author's Name, Title, Conference/Journal, Abstract, Year,
Algorithm, Link)

**Method:** Every row's `Algorithm`, `Title`, and `Abstract` fields were
programmatically searched (case-insensitive) for the strings "mamba",
"state space", "selective ssm", and " ssm". Zero matches were found across
all 25 rows. The full algorithm list per paper is reproduced below for
transparency.

## Algorithms identified in the existing literature set (all 25 papers)

| # | Year | Title | Algorithm(s) listed |
|---|------|-------|----------------------|
| 1 | 2025 | Adaptive Knowledge Distillation for Device-Directed Speech Detection | Adaptive Knowledge Distillation, Task-Specific Adapters, Transformer, Conformer |
| 2 | 2025 | PruneSLU | Vocabulary and Structural Pruning, Integration Knowledge Distillation, Contrastive Learning |
| 3 | 2025 | SOVA-Bench | Speech Large Language Models, Speech Encoders, Neural Vocoders (benchmark) |
| 4 | 2025 | Joint speech and text machine translation for up to 100 languages | Unified Multitask Seq2Seq Transformer |
| 5 | 2025 | VoxEval | End-to-End Spoken Language Models, SpeechQA Benchmarking |
| 6 | 2025 | Distilling an End-to-End Voice Assistant | Cross-Modal Knowledge Distillation, Self-Supervision, Speech LLM |
| 7 | 2025 | WavRAG | Retrieval-Augmented Generation, Joint Audio–Text Embedding Retrieval, CoT, LLM |
| 8 | 2024 | SILENCE | Differentiable Mask Generation, Interpretable Learning, Disentanglement-Based Speech Encoding |
| 9 | 2024 | A Full-duplex Speech Dialogue Scheme Based On LLM | LLM, Neural Finite State Machine, Next-Token Prediction |
| 10 | 2024 | DiscreteSLU | LLM, Self-Supervised Discrete Speech Units, k-means, Speech Adapter, Instruction Tuning |
| 11 | 2024 | MM-KWS | Multi-modal Prompt Embeddings, Multilingual Pre-trained Encoders, Hard-Negative Augmentation |
| 12 | 2024 | Scaling Speech Technology to 1,000+ Languages | wav2vec 2.0, Self-Supervised Learning, CTC Multilingual ASR |
| 13 | 2024 | OWSM-CTC | CTC, Encoder-Only Speech Foundation Model, Non-Autoregressive Multitask Learning |
| 14 | 2024 | AudioGPT | LLM (ChatGPT) orchestrating Audio Foundation Models, ASR/TTS Interface |
| 15 | 2023 | SpokenWOZ | Dual-Modal Transformer Models, LLMs, Dialogue State Tracking |
| 16 | 2023 | Multi-task deep cross-attention networks | Multi-Task Learning, Deep Cross-Attention Network, Shared Encoder, Soft Attention |
| 17 | 2023 | SpeechGPT | LLM, Discrete Speech Representations, Cross-Modal Instruction Fine-Tuning |
| 18 | 2023 | Personalized Predictive ASR | Streaming End-to-End ASR, Predictive Utterance Modelling, Speaker-Level Personalization |
| 19 | 2023 | Conmer | Conformer (self-attention-free variant), CNN, Streaming Neural Transducer |
| 20 | 2023 | Efficient Multimodal Neural Networks for Trigger-less Voice Assistants | Multimodal Audio–Gesture Fusion Network, Lightweight On-Device DNN |
| 21 | 2023 | Whisper (Robust Speech Recognition via Large-Scale Weak Supervision) | Transformer (Encoder–Decoder), Large-Scale Weakly Supervised Multitask Learning |
| 22 | 2022 | Production federated keyword spotting | Federated Learning, Knowledge Distillation, Confidence Filtering |
| 23 | 2022 | Device-Directed Speech Detection via Distillation | Knowledge Distillation, LatticeRNN, Weakly-Supervised Acoustic Modelling, Ensemble |
| 24 | 2022 | Self-supervised learning with random-projection quantizer (BEST-RQ) | Self-Supervised Learning, Random-Projection Quantizer, Masked Prediction |
| 25 | 2022 | Branchformer | Parallel Self-Attention + cgMLP branches, Transformer, Conformer |

## Algorithm family summary

The 25 papers cluster into these algorithmic families:

- **Transformer / Conformer / Branchformer** self-attention-based sequence
  encoders (papers 1, 4, 15, 19, 21, 25)
- **Large Language Model (LLM)–centric** speech/voice-assistant systems,
  including speech-LLM integration, RAG, and instruction tuning (papers
  3, 5, 6, 7, 9, 10, 14, 15, 17)
- **Knowledge distillation / model compression** (papers 1, 2, 6, 22, 23)
- **Self-supervised representation learning** (wav2vec 2.0, BEST-RQ)
  (papers 12, 24)
- **CTC / non-autoregressive foundation models** (paper 13)
- **Federated learning** (paper 22)
- **Multimodal fusion (audio + gesture / text)** (papers 4, 7, 20)
- **Recurrent components used only as an auxiliary/lattice element**, not as
  the principal sequence model (paper 23's "LatticeRNN" is a decoding
  lattice-scoring component, not the encoder backbone)

## Verification: Mamba / Selective SSM absence

**Result: Confirmed absent.** No paper in the 25-paper set lists Mamba,
Selective State Space Model, S4, S5, or any state-space-model variant as its
algorithm. No title or abstract contains "mamba," "state space," or "SSM."
The closest architectural family present is Transformer/Conformer
(self-attention-based), which Mamba is explicitly positioned as an
alternative to — not a variant of.

## Why Mamba is a distinct algorithmic approach

Mamba (Gu & Dao, 2023) is a Selective State Space Model: it processes a
sequence via a continuous-time-inspired linear recurrence with
input-dependent ("selective") parameters, achieving linear-time sequence
scaling. This is mechanistically different from every algorithm family
represented in the existing set:

- Unlike **self-attention** (Transformer/Conformer/Branchformer), Mamba does
  not compute pairwise token interactions; it maintains a compressed
  recurrent hidden state updated token-by-token, which is why it scales
  linearly rather than quadratically with sequence length.
- Unlike a standard **RNN/GRU/LSTM**, Mamba's state-transition and input
  matrices are *input-dependent* (selective) rather than fixed per layer,
  which is the core technical contribution distinguishing it from classical
  recurrence (and from paper 23's LatticeRNN).
- It is not a knowledge-distillation, federated-learning, or
  self-supervised-pretraining method — those are training paradigms
  applied on top of some sequence model, not sequence-modeling
  architectures themselves, and are orthogonal to this audit.

## Relevant Mamba papers (to be consulted during methodology write-up)

- Gu, A. & Dao, T. (2023/2024). *Mamba: Linear-Time Sequence Modeling with
  Selective State Spaces.* (arXiv:2312.00752)
- Zhang, X. et al. *Mamba in Speech: Towards an Alternative to
  Self-Attention.* — speech-domain application of Mamba, directly relevant
  to positioning this project's contribution.

(Full bibliographic verification and additional Speech-Mamba/Mamba-ASR
literature will be recorded separately once fetched from primary sources;
this audit's purpose is the absence check above, not the full literature
review.)

## Positioning relative to existing literature

This project's contribution is an **experimental evaluation and
implementation of compact Mamba-based speech-command recognition on Apple
Silicon**, positioned as introducing a fundamentally different
sequence-modeling paradigm (selective SSM) into a literature set that is
otherwise entirely self-attention-based (Transformer/Conformer family) or
LLM-orchestration-based, with recurrence appearing only as a minor
auxiliary component (LatticeRNN) rather than as a modern selective-SSM
backbone. No overlapping or redundant algorithm exists in the reviewed set.

---
*Audit performed programmatically against the source Excel file on
2026-08-24. Method: exact-match keyword search (case-insensitive) across
Algorithm, Title, and Abstract columns for the terms "mamba", "state space",
"selective ssm", "ssm".*
