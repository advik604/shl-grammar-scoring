# Grammar Scoring from Audio (SHL Kaggle Hiring Assessment 2026)

This repository contains a complete, end-to-end Machine Learning pipeline to predict a continuous grammar score (0–5, MOS Likert scale) from 45–60 second English speech audio (`.wav`) files.

## Project Overview

The objective of this challenge is to assess non-native English speakers' grammatical proficiency from raw speech recordings. The target variable is a continuous Mean Opinion Score (MOS) ranging from 0.0 to 5.0 (in increments of 0.5, with severe disfluencies/degraded clips anchored at 0.0). Evaluation metric is **Root Mean Squared Error (RMSE)** (lower is better).

### Repository Structure

```
grammar-scoring/
├── notebooks/
│   ├── 01_transcribe.ipynb   # Audio EDA, verbatim transcription & fluency feature extraction via faster-whisper
│   └── 02_model.ipynb        # Multi-encoder audio suite, layer band selection, SVR/Ridge CV & nested calibration
├── README.md                 # Project overview, results progression, and reproduction guide
├── .gitignore                # Git ignore rules ensuring dataset and cache binaries are excluded
└── requirements.txt          # Repository dependencies reference (documentation only)
```

> [!NOTE]
> **Data Privacy Notice**: Per competition guidelines, dataset audio files (`.wav`), CSV files (`train.csv`, `test.csv`, `sample_submission.csv`), and generated embedding cache files (`*.npy`, `transcripts.csv`) are strictly excluded from version control via `.gitignore`.

---

## Results Progression Table

| Pipeline Version | Architectural Highlights | OOF CV RMSE | OOF Pearson $r$ | In-Sample Train RMSE | In-Sample Pearson $r$ | Public Leaderboard RMSE |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **v1 Baseline** | Verbatim transcription text features + RoBERTa-CoLA + sentence-transformers | 0.799 | 0.772 | 0.533 | — | 0.5543 |
| **v2 Hybrid** | Text features + `WavLM-base-plus` (Layer 10) + LightGBM / SVR / Ridge | 0.545 | 0.901 | 0.486 | — | 0.4085 |
| **v3 Full Suite** | Multi-Encoder (`WavLM-base`, `WavLM-large`, `Whisper-large-v3` encoder) + Layer Band Selection + SVR/Ridge + NNLS Blend + Nested Linear Calibration | **0.491** | **0.918** | **0.192** | **0.988** | **0.4096** |

*Note: In-sample training RMSE for v3 is 0.192 (Pearson 0.988), evaluated via full pipeline refit. Flexible models (SVR, LightGBM) fit the training data almost perfectly, so Out-of-Fold (OOF) RMSE (0.491, Pearson 0.918) is the honest estimate.*

---

## Technical Architecture (v3 Pipeline)

### Phase 1: Verbatim Transcription & Cadence Extraction (`01_transcribe.ipynb`)
- **PyAV Mismatch Fix & `soundfile` Bypass**: Reads audio via `soundfile` as 16 kHz mono `float32` arrays directly, passing raw NumPy arrays to `faster-whisper` (`large-v3`) to bypass PyAV version conflicts on Kaggle.
- **Disfluency Prompt**: Uses `initial_prompt="Um, uh, like, you know..."` to force verbatim capture of stutters, filler words, and disfluencies.
- **Fail-Safe Exception Handling**: Captures audio processing failures (e.g. `audio_709.wav`) in `failed_files.csv` while persisting valid transcriptions to `transcripts.csv`.
- **Acoustic Cadence Features**: Extracts `speech_rate` (words/sec), pause count (`num_pauses_05`, gaps $>0.5$s), mean pause duration, segment logprobs (`mean_avg_logprob`), and `mean_no_speech_prob`.

### Phase 2: Feature Engineering & Multi-Encoder Speech Suite (`02_model.ipynb`)
1. **Disfluency Prompt Stripping**: Cleanly removes initial prompt leakage before computing text metrics.
2. **Group A (Rate & Cadence Features)**: Filler word rates, repetition rates, Type-Token Ratio (TTR), sentence length stats, complex syntax marker rates, and speech rates. *Note: Raw length features (`duration_sec` and `num_words`) are explicitly dropped to prevent train/test length shift bias.*
3. **Group B (Grammatical Acceptability)**: Zero-shot sentence acceptability scoring via `textattack/roberta-base-CoLA`.
4. **Group C (Text Embeddings)**: Sentence embeddings via `sentence-transformers/all-mpnet-base-v2`.
5. **Group D (Compression & Hallucination)**: `gzip` compression ratio, 3-gram/4-gram repetition rates, unique sentence ratios, prompt leakage flags, and Whisper confidence.
6. **Group E (Multi-Encoder Audio Suite)**:
   - `microsoft/wavlm-base-plus` (13 layers, hidden dim 768)
   - `microsoft/wavlm-large` (25 layers, hidden dim 1024, fp16 autocast)
   - `openai/whisper-large-v3` encoder (33 layers, hidden dim 1280, padded frame masking)
7. **Layer Band Window Selection**: Evaluates candidate 3–5 layer windows across all hidden states to select optimal depth bands (upper layers capture speech quality best: `WavLM-base` layers 7–10/12, `WavLM-large` layers 20–23/24, `Whisper encoder` layers 24–28/32).
8. **Multi-Seed Leak-Free Cross-Validation**:
   - 5-fold Stratified K-Fold CV (binned continuous target) across 3 random seeds (`[42, 101, 2024]`).
   - Standard scaling, PCA dimension reduction (for high-dimensional audio features), and alpha hyperparameter grids are strictly fit within each fold pipeline to prevent data leakage.
   - Evaluates Ridge Regressors, Non-linear Support Vector Regression (SVR with RBF kernel), LightGBM, and Non-Negative Least Squares (NNLS) ensembling.
9. **Nested Linear Calibration**: Fits an out-of-fold linear calibration mapping evaluated via nested cross-validation.

---

## Kaggle Reproduction Guide

To run this project on Kaggle GPU T4 environments:

### Step 1: Execute `notebooks/01_transcribe.ipynb`
1. Create a new Kaggle Notebook with GPU T4 accelerator enabled.
2. Attach dataset: `/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final`.
3. Set `QUICK_TEST = False` in the configuration cell for full execution (or `True` for a 10-file sanity check).
4. Run all cells. The notebook will output `transcripts.csv` and `failed_files.csv` in `/kaggle/working/`.

### Step 2: Execute `notebooks/02_model.ipynb`
1. Create a second Kaggle Notebook with GPU T4 accelerator enabled.
2. Attach dataset `/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final`.
3. Add the output of Notebook 01 as a Kaggle Notebook Dataset dependency so that `transcripts.csv` is located at `/kaggle/input/.../transcripts.csv`.
4. Run all cells. The notebook automatically detects `transcripts.csv`, extracts audio features, performs layer band selection, trains cross-validated models, and exports `/kaggle/working/submission.csv`.

### Step 3: Verify Submission File
The script automatically asserts:
- Output file exists at `/kaggle/working/submission.csv`.
- Exactly **216 rows** matching `test.csv` order.
- Columns `filename` and `label`.
- All predicted values bounded strictly in $[0.0, 5.0]$ with zero missing/NaN entries.

---

## Key Technical Findings

1. **Audio Embeddings > Text Transcripts**: Audio representations carry dramatically higher predictive power than transcripts alone (audio-only models yield 0.523–0.567 OOF RMSE vs 0.865–0.883 for text-only models in v3). Speech quality, rhythm, and accent nuances are captured in the acoustic signal lost in text transcription.
2. **Layer Band Depth Sensitivity**: Upper acoustic hidden states (`WavLM-base` layers 7–10/12, `WavLM-large` layers 20–23/24, `Whisper encoder` layers 24–28/32) deliver peak performance. Lower acoustic layers preserve raw audio wave details, while final layers encode ASR token probability distributions.
3. **Whisper Logprob Significance**: Whisper's segment logprob (`mean_avg_logprob`, $r = 0.606$) is the single most informative textual feature, reflecting model acoustic decoding confidence.
4. **Length Bias Prevention**: Test clips are shorter than training clips (mean 48.8s vs 55.8s). Removing raw count features (`duration_sec`, `num_words`) while preserving rate-based metrics eliminated train/test distribution drift.
5. **Score 0.0 Disfluency Anchors**: Samples labeled 0.0 represent severe disfluencies or corrupted audio. Retaining them in stratified CV provides crucial anchor points for bounded prediction calibration.
6. **CV vs. Leaderboard Insights**: The v3 architecture improved OOF RMSE from 0.545 to 0.491 (Pearson $r$ improved from 0.901 to 0.918). We hypothesize that this occurs because v3 gains concentrated on extreme low/high scores and edge cases (such as label 0.0), which are rare in the small public test split (~130 clips), while maintaining strong overall stability.
