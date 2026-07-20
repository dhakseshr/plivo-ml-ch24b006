# End-of-Turn Detection — plivo-ml-ch24b006

Predicts `p_eot` (probability the user has finished their turn) at every silence pause in a voice conversation, enabling a voice AI agent to respond with minimal delay and minimal interruptions.

## Results

| Language | Mean Response Delay | Interrupted Turns | AUC (OOF) |
|----------|--------------------|--------------------|-----------|
| English  | 1255 ms            | 5.0%               | 0.688     |
| Hindi    | 745 ms             | 5.0%               | 0.804     |
| Baseline (silence-only) | ~1600 ms | — | — |

## How to Run

```bash
python predict.py --data_dir eot_data/english --out predictions_en.csv
python predict.py --data_dir eot_data/hindi   --out predictions_hi.csv
```

The `data_dir` must contain `labels.csv` and `audio/<turn_id>.wav`. Output columns: `turn_id, pause_index, p_eot`.

## Approach

### Features (causal — only audio before `pause_start`)

- **F0 / pitch**: slope, final/mean ratio, speaker-normalized z-score, voiced fraction, last voiced region duration, fall composite — falling intonation signals statement completion
- **Energy**: slope, decay ratio, end-vs-start ratio over last 0.5s window
- **MFCC**: 13 coefficient means + 13 delta means from last 0.5s — spectral dynamics near the pause boundary
- **Spectral**: centroid, rolloff, flux slope/final/mean — spectral energy shifts lower at EOT
- **Turn context**: pause index, pause_start, inter-pause interval, n_prior pauses, language flag

### Models

**GBM + RF Ensemble (weight 0.7)**
- Soft-voting of `GradientBoostingClassifier` (200 trees, depth 3) and `RandomForestClassifier` (300 trees, depth 6)
- Trained on English + Hindi combined; isotonic probability calibration
- 5-fold GroupKFold cross-validation (by turn) for OOF evaluation

**Bidirectional LSTM (weight 0.3)**
- Frame-level features at 10ms hop: F0, energy, 13 MFCCs → 80-frame sequence window
- Bidirectional LSTM with attention pooling + scalar feature branch
- Focal loss to handle class imbalance; OneCycleLR scheduler

Final `p_eot = 0.7 × ensemble_prob + 0.3 × lstm_prob`

## Iteration Log

| Run | Change | Hindi delay | CV AUC |
|-----|--------|------------|--------|
| 1 | Silence-only baseline | ~1600ms | — |
| 2 | Prosodic features + GBM | — | 0.604 (val) |
| 3 | + MFCC + spectral centroid | — | 0.736 (val) |
| 4 | GBM+RF ensemble + OOF eval | 843ms | 0.701 |
| 5 | Error analysis: narrowed windows, speaker-norm F0, language flag | 781ms | 0.728 |
| **6** | **+ LSTM blend (final)** | **745ms** | **0.728** |

See `RUNLOG.md` for full details per run.

## File Structure

```
├── predict.py            # Inference: loads both models, blends, writes predictions.csv
├── train_model.py        # Trains GBM+RF ensemble with 5-fold OOF → model.pkl
├── train_lstm.py         # Trains Bidirectional LSTM → model_lstm.pt
├── feature_extraction.py # 56 scalar causal features
├── frame_features.py     # Frame-level feature sequence for LSTM
├── model_lstm.py         # LSTM architecture (EOTClassifier)
├── error_analysis.py     # OOF error analysis used to guide Run 5
├── model.pkl             # Trained ensemble (all data)
├── model_lstm.pt         # Trained LSTM checkpoint
├── predictions.csv       # Final predictions (English + Hindi combined)
├── predictions_en.csv    # English predictions
├── predictions_hi.csv    # Hindi predictions
├── ensemble_en.csv       # Ensemble-only predictions (English)
├── ensemble_hi.csv       # Ensemble-only predictions (Hindi)
├── RUNLOG.md             # Per-run scores and changes
├── NOTES.md              # Model signals, failure modes, future work
└── SUMMARY.html          # Full writeup with graphs and analysis
```

## Known Failure Modes

- **Flat declaratives**: Hindi dialects with level pitch on statements confuse F0 features
- **Short turns** (< 1s speech): insufficient context for reliable F0 estimation
- **List enumerations**: EOT-like acoustics between items, speaker continues
- **Speaker variability**: no learned speaker normalization across callers
