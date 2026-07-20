"""
End-of-Turn prediction using a pre-trained model.

Usage:
    python predict.py --data_dir <folder> --out predictions.csv

The folder must have the same structure as the training data:
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import extract_features, load_wav

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found at {MODEL_PATH}. Run train_model.py first.")

    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    labels_path = os.path.join(args.data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))

    wav_cache = {}
    turn_pauses = {}
    X, keys = [], []

    for r in rows:
        tid = r["turn_id"]
        pause_start = float(r["pause_start"])
        pause_index = int(r["pause_index"])

        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in wav_cache:
            wav_cache[path] = load_wav(path)
        x, sr = wav_cache[path]

        lang = "hi" if "hindi" in args.data_dir.lower() else "en"
        prior = turn_pauses.get(tid, [])
        feat = extract_features(x, sr, pause_start, pause_index, prior, lang=lang)
        turn_pauses.setdefault(tid, []).append(pause_start)

        X.append(feat)
        keys.append((tid, r["pause_index"]))

    X = np.array(X)
    p = pipeline.predict_proba(X)[:, 1]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), prob in zip(keys, p):
            w.writerow([tid, pi, f"{prob:.4f}"])

    print(f"Wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
