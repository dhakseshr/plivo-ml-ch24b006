"""
Train the EOT classifier on provided data (English + Hindi combined),
save the model to model.pkl, and write OOF cross-val predictions.csv for scoring.

Usage:
    python train_model.py --data_dirs eot_data/english eot_data/hindi \
                          --out_en predictions_en.csv \
                          --out_hi predictions_hi.csv
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_features, load_wav, FEATURE_NAMES


def load_dataset(data_dir):
    labels_path = os.path.join(data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))
    wav_cache = {}
    X, y, groups, keys = [], [], [], []
    turn_pauses = {}
    for r in rows:
        tid = r["turn_id"]
        pause_start = float(r["pause_start"])
        path = os.path.join(data_dir, r["audio_file"])
        if path not in wav_cache:
            wav_cache[path] = load_wav(path)
        x, sr = wav_cache[path]
        lang = "hi" if "hindi" in data_dir else "en"
        prior = turn_pauses.get(tid, [])
        feat = extract_features(x, sr, pause_start, int(r["pause_index"]), prior, lang=lang)
        turn_pauses.setdefault(tid, []).append(pause_start)
        X.append(feat); y.append(1 if r["label"] == "eot" else 0)
        groups.append(tid); keys.append((tid, r["pause_index"], r["audio_file"]))
    return np.array(X), np.array(y), groups, keys


def build_pipeline():
    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.8, min_samples_leaf=8, random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    # Soft-voting ensemble
    ensemble = VotingClassifier(
        estimators=[("gbm", gbm), ("rf", rf)],
        voting="soft",
        weights=[1, 1],
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(ensemble, cv=3, method="isotonic")),
    ])


def auc_score(y_true, scores):
    y_true = np.array(y_true)
    scores = np.array(scores)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1, n0 = y_true.sum(), len(y_true) - y_true.sum()
    return (ranks[y_true == 1].sum() - n1*(n1+1)/2) / (n1*n0) if n1 and n0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--out_en", default="predictions_en.csv")
    ap.add_argument("--out_hi", default="predictions_hi.csv")
    ap.add_argument("--model_out", default="model.pkl")
    args = ap.parse_args()

    per_dir = {}
    all_X, all_y, all_groups, all_keys, all_dirs = [], [], [], [], []

    for d in args.data_dirs:
        print(f"Loading {d} ...")
        X, y, groups, keys = load_dataset(d)
        per_dir[d] = (X, y, groups, keys)
        all_X.append(X); all_y.append(y)
        all_groups.extend(groups); all_keys.extend(keys)
        all_dirs.extend([d] * len(y))

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    print(f"\nCombined: {len(y_all)} pauses, {y_all.sum()} EOT ({100*y_all.mean():.1f}%)")

    # 5-fold group cross-val OOF predictions (honest evaluation)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y_all))
    for fold, (tr, te) in enumerate(gkf.split(X_all, y_all, all_groups)):
        p = build_pipeline()
        p.fit(X_all[tr], y_all[tr])
        oof[te] = p.predict_proba(X_all[te])[:, 1]
        print(f"  Fold {fold+1} AUC: {auc_score(y_all[te], oof[te]):.3f}")

    print(f"\n5-fold CV AUC: {auc_score(y_all, oof):.3f}")

    # Final model trained on ALL data
    print("Training final model on all data ...")
    final_pipeline = build_pipeline()
    final_pipeline.fit(X_all, y_all)
    with open(args.model_out, "wb") as f:
        pickle.dump(final_pipeline, f)
    print(f"Model saved -> {args.model_out}")

    # Feature importances from GBM inside the calibrated ensemble
    try:
        inner = final_pipeline.named_steps["clf"].calibrated_classifiers_[0].estimator
        gbm_est = inner.named_estimators_["gbm"]
        importances = gbm_est.feature_importances_
        top = np.argsort(importances)[::-1][:8]
        print("Top GBM features:", [(FEATURE_NAMES[i], f"{importances[i]:.3f}") for i in top])
    except Exception as e:
        print(f"(Feature importance unavailable: {e})")

    # Write predictions using OOF scores (honest cross-val)
    def write_preds(data_dir, out_path):
        _, _, _, keys_d = per_dir[data_dir]
        dir_mask = [i for i, d in enumerate(all_dirs) if d == data_dir]
        oof_d = oof[dir_mask]
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["turn_id", "pause_index", "p_eot"])
            for (tid, pi, _), p in zip(keys_d, oof_d):
                w.writerow([tid, pi, f"{p:.4f}"])
        print(f"Wrote {len(keys_d)} OOF predictions -> {out_path}")

    for d in args.data_dirs:
        out = args.out_en if "english" in d else args.out_hi
        write_preds(d, out)


if __name__ == "__main__":
    main()
