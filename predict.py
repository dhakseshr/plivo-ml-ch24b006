"""
End-of-Turn prediction using blended ensemble + LSTM model.

Usage:
    python predict.py --data_dir <folder> --out predictions.csv

The folder must have the same structure as training data:
    <data_dir>/
        labels.csv
        audio/<turn_id>.wav
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import extract_features, load_wav
from frame_features import extract_frame_sequence, N_FRAME_FEATS, SR, HOP
from train_lstm import scalar_features, N_SCALAR
from model_lstm import EOTClassifier
from frame_features import SEQ_LEN

ENSEMBLE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
LSTM_MODEL     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_lstm.pt")
ENSEMBLE_W = 0.7   # optimal blend weight found via grid search
LSTM_W     = 0.3


def load_models():
    models = {}
    if os.path.exists(ENSEMBLE_MODEL):
        with open(ENSEMBLE_MODEL, "rb") as f:
            models["ensemble"] = pickle.load(f)
    if os.path.exists(LSTM_MODEL):
        ckpt = torch.load(LSTM_MODEL, map_location="cpu")
        lstm = EOTClassifier(
            n_frame_feats=ckpt.get("n_frame_feats", N_FRAME_FEATS),
            n_scalar_feats=ckpt.get("n_scalar_feats", N_SCALAR),
        )
        lstm.load_state_dict(ckpt["model_state"])
        lstm.eval()
        models["lstm"] = lstm
    if not models:
        sys.exit("No model files found. Run train_model.py and train_lstm.py first.")
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    models = load_models()
    lang = "hi" if "hindi" in args.data_dir.lower() else "en"

    labels_path = os.path.join(args.data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))

    wav_cache = {}
    turn_pauses = {}
    turn_stats  = {}

    ens_feats, lstm_seqs, lstm_masks, lstm_scs, keys = [], [], [], [], []

    for r in tqdm(rows, desc="extracting features", unit="pause"):
        tid = r["turn_id"]
        pause_start  = float(r["pause_start"])
        pause_index  = int(r["pause_index"])

        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in wav_cache:
            wav_cache[path] = load_wav(path)
        x, sr = wav_cache[path]

        prior = turn_pauses.get(tid, [])

        # ensemble features
        ens_feat = extract_features(x, sr, pause_start, pause_index, prior, lang=lang)
        ens_feats.append(ens_feat)

        # lstm features
        spk_stats = turn_stats.get(tid, None)
        seq, mask, spk_stats_out = extract_frame_sequence(x, sr, pause_start, spk_stats)
        if tid not in turn_stats:
            turn_stats[tid] = spk_stats_out
        sc = scalar_features(pause_start, pause_index, prior, seq, mask)
        lstm_seqs.append(seq)
        lstm_masks.append(mask)
        lstm_scs.append(sc)

        turn_pauses.setdefault(tid, []).append(pause_start)
        keys.append((tid, r["pause_index"]))

    # ensemble predictions
    p_ens = np.zeros(len(rows))
    if "ensemble" in models:
        X = np.array(ens_feats)
        p_ens = models["ensemble"].predict_proba(X)[:, 1]

    # lstm predictions
    p_lstm = np.zeros(len(rows))
    if "lstm" in models:
        seqs  = torch.tensor(np.array(lstm_seqs),  dtype=torch.float32)
        masks = torch.tensor(np.array(lstm_masks),  dtype=torch.bool)
        scs   = torch.tensor(np.array(lstm_scs),    dtype=torch.float32)
        with torch.no_grad():
            logits = models["lstm"](seqs, masks, scs)
            p_lstm = torch.sigmoid(logits).numpy()

    # blend
    if "ensemble" in models and "lstm" in models:
        p_final = ENSEMBLE_W * p_ens + LSTM_W * p_lstm
    elif "ensemble" in models:
        p_final = p_ens
    else:
        p_final = p_lstm

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), p in zip(keys, p_final):
            w.writerow([tid, pi, f"{p:.4f}"])

    print(f"Wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
