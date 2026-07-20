# RUNLOG

## Run 6 — LSTM blend (final submission)
**Score (English):** 1255 ms @ 5.0% interrupted | AUC 0.688 (OOF)
**Score (Hindi):** 745 ms @ 5.0% interrupted | AUC 0.804 (OOF)
**Change:** Added Bidirectional LSTM with attention pooling trained on frame-level features (F0, energy, MFCC at 10ms hop, 80-frame window). Blended with GBM+RF ensemble at 0.7/0.3 weight (grid-searched). predict.py updated to load both models and blend probabilities at inference time.
**Why:** Frame-level sequential model captures intonation contour dynamics that scalar features cannot — the LSTM sees the full pitch trajectory over the last 800ms, not just slope/ratio summaries. Blend outperforms either model alone on Hindi.

---

## Run 5 — Error analysis + targeted feature redesign (CV AUC 0.728)
**Score (English):** 1239 ms @ 5.0% interrupted | AUC 0.682 (OOF)
**Score (Hindi):** 781 ms @ 4.0% interrupted | AUC 0.776 (OOF)
**5-fold CV AUC:** 0.728 (up from 0.701)
**Change:** Full error analysis on OOF predictions revealed: (1) first-pause EOTs (pause_index=0) systematically under-predicted — model lacked first-pause priors; (2) Hindi HOLD pauses over-predicted as EOT; (3) 1.5s window was diluting end-of-utterance signal by averaging over too much context. Fixes: narrowed energy and F0 slope features to last 0.5s window, added speaker-normalized F0 (z-score within turn), added explicit language flag, added n_prior pauses and avg inter-pause interval, switched all F0 to pyin throughout. Refocused spectral features to 0.5s window.
**Why:** Feedback from scoring analysis: optimizing AUC is necessary but the real signal comes from reducing the gap between EOT p_eot distribution and HOLD p_eot distribution specifically at the 5% false-cutoff operating point. Shorter window captures the decisive boundary signal.

---

## Run 1 — Silence-only baseline
**Score (English):** 1600 ms @ 0.0% interrupted  
**Change:** Reference baseline from `starter/baseline.py` — predicts p_eot=1.0 for every pause.  
**Why:** Establishes floor; agent waits full 1.6 s timeout for every turn.

---

## Run 2 — Prosodic features + GradientBoostingClassifier (English+Hindi combined)
**Score (English):** 160 ms @ 1.0% interrupted | AUC 0.999  
**Score (Hindi):** 100 ms @ 5.0% interrupted | AUC 1.000  
**Val AUC (held-out turns):** 0.604  
**Change:** Replaced starter's 3 weak features with 20 prosodic+temporal features: F0 slope, F0 final/mean ratio, energy slope, energy decay ratio, voiced fraction, last voiced region duration (syllable lengthening proxy), ZCR, pause index, pause_start, inter-pause interval. Trained GradientBoostingClassifier (300 trees, depth 4) with isotonic calibration. Trained on both languages combined.  
**Why:** Prosodic features (falling pitch = statement = EOT, falling energy, syllable lengthening) are the linguistically motivated signals for turn-completion; combined training exploits the shared phonological patterns across English and Hindi.

---

## Run 4 — Add spectral rolloff + flux; switch to GBM+RF ensemble; use OOF predictions
**Score (English):** 1313 ms @ 5.0% interrupted | AUC 0.683 (OOF)
**Score (Hindi):** 843 ms @ 5.0% interrupted | AUC 0.730 (OOF)
**5-fold CV AUC:** 0.701
**Change:** Added spectral rolloff slope/final/mean and spectral flux features (total 56 features). Replaced single GBM with soft-voting ensemble of GBM (200 trees, depth 3) + RandomForest (300 trees, depth 6). Switched predictions.csv from in-sample (inflated) to OOF cross-validation scores for honest evaluation.
**Why:** OOF predictions give a realistic picture of hidden-test performance. The higher delay vs run 3 reflects honest generalization, not overfitting. 5-fold CV AUC of 0.701 is a stable estimate. model.pkl is still trained on all data for best generalization on the hidden test.

---

## Run 3 — MFCC + spectral centroid features added
**Score (English):** 100 ms @ 5.0% interrupted | AUC 1.000  
**Score (Hindi):** 100 ms @ 0.0% interrupted | AUC 1.000  
**Val AUC (held-out turns):** 0.736  
**Change:** Added 13 MFCC means, 13 MFCC delta means (spectral dynamics over last 0.5 s), spectral centroid slope/mean/final, and local 300 ms energy slope. Total 50 features.  
**Why:** Val AUC jumped from 0.604 → 0.736. MFCC deltas capture rate-of-change in vocal tract configuration; mfcc_delta_10 and mfcc_10 emerged as top features alongside f0_final_ratio, suggesting spectral dynamics near the pause boundary carry EOT signal beyond pitch alone. This improves expected generalization on the hidden Hindi test set.
