"""
Train the LSTM EOT classifier.

Strategy:
- 5-fold GroupKFold (turns never split across train/val)
- Focal loss to handle class imbalance
- Cosine LR schedule with warmup
- Save best checkpoint per fold, ensemble OOF for predictions.csv
- Final model: retrain on ALL data for model_lstm.pt (used by predict.py)

Usage:
    python train_lstm.py \
        --data_dirs eot_handout/eot_data/english eot_handout/eot_data/hindi \
        --out_en predictions_en.csv \
        --out_hi predictions_hi.csv \
        --model_out model_lstm.pt
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from frame_features import load_wav, extract_frame_sequence, SEQ_LEN, N_FRAME_FEATS, SR, HOP
from model_lstm import EOTClassifier
from feature_extraction import extract_features  # scalar features from ensemble model

DEVICE = "cpu"
N_SCALAR = 8
EPOCHS = 60
BATCH = 16
LR = 3e-3
SEED = 42


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def scalar_features(pause_start, pause_index, prior_pauses, seq, mask):
    """
    8 scalar features derived from already-extracted frame sequence.
    No audio re-processing — avoids double pyin call.

    seq : (SEQ_LEN, N_FRAME_FEATS) — col 1 = norm F0, col 2 = voiced flag
    mask: (SEQ_LEN,) bool
    """
    hop_s = HOP / SR

    time_since_last = pause_start - prior_pauses[-1] if prior_pauses else pause_start
    seg15_len = min(pause_start, 1.5)

    # derive voiced frac and F0 slope from the frame sequence (already computed)
    real_frames = seq[mask]   # (n_real, F)
    if len(real_frames) > 0:
        voiced_flag = real_frames[:, 2]              # col 2 = voiced flag
        voiced_frac = float(voiced_flag.mean())
        f0_norm = real_frames[:, 1]                  # col 1 = normalised F0
        voiced_pos = np.where(voiced_flag > 0.5)[0]
        if len(voiced_pos) >= 4:
            tail = min(8, len(voiced_pos))
            f0_slope = float(np.polyfit(
                voiced_pos[-tail:], f0_norm[voiced_pos[-tail:]], 1)[0])
        else:
            f0_slope = 0.0
    else:
        voiced_frac = 0.0
        f0_slope = 0.0

    return np.array([
        float(pause_index),
        pause_start,
        seg15_len,
        time_since_last,
        float(len(prior_pauses)),
        voiced_frac,
        f0_slope,
        1.0 if pause_index == 0 else 0.0,
    ], dtype=np.float32)


class EOTDataset(Dataset):
    def __init__(self, records):
        # records: list of (seq, mask, scalar, label)
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        seq, mask, sc, label = self.records[idx]
        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(sc, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def load_dataset(data_dir):
    labels_path = os.path.join(data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))
    wav_cache = {}
    records, groups, keys = [], [], []
    turn_pauses = {}
    turn_stats = {}   # speaker stats per turn (computed from first pause)

    for r in tqdm(rows, desc=f"  extracting {os.path.basename(data_dir)}", unit="pause"):
        tid = r["turn_id"]
        pause_start = float(r["pause_start"])
        pause_index = int(r["pause_index"])
        label = 1 if r["label"] == "eot" else 0

        path = os.path.join(data_dir, r["audio_file"])
        if path not in wav_cache:
            wav_cache[path] = load_wav(path)
        x, sr = wav_cache[path]

        prior = turn_pauses.get(tid, [])
        speaker_stats = turn_stats.get(tid, None)

        seq, mask, spk_stats = extract_frame_sequence(x, sr, pause_start,
                                                       speaker_stats)
        if tid not in turn_stats:
            turn_stats[tid] = spk_stats

        # scalar features derived from seq — no second pyin call
        sc = scalar_features(pause_start, pause_index, prior, seq, mask)
        turn_pauses.setdefault(tid, []).append(pause_start)

        records.append((seq, mask, sc, label))
        groups.append(tid)
        keys.append((tid, r["pause_index"], r["audio_file"], data_dir))

    return records, groups, keys


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=1.5):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor(self.pos_weight),
            reduction="none"
        )
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def auc_score(y_true, scores):
    y_true = np.array(y_true)
    scores = np.array(scores)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1, n0 = y_true.sum(), len(y_true) - y_true.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def train_one_fold(train_recs, val_recs, epochs=EPOCHS):
    set_seed(SEED)
    model = EOTClassifier(n_frame_feats=N_FRAME_FEATS, n_scalar_feats=N_SCALAR).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=epochs * max(1, len(train_recs) // BATCH),
        pct_start=0.2, anneal_strategy="cos",
    )
    criterion = FocalLoss(gamma=2.0, pos_weight=1.8)

    train_loader = DataLoader(EOTDataset(train_recs), batch_size=BATCH,
                              shuffle=True, drop_last=False)
    best_auc = 0.0
    best_state = None

    pbar = tqdm(range(epochs), desc="  training", unit="epoch", leave=False)
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        for seq, mask, sc, labels in train_loader:
            optimizer.zero_grad()
            logits = model(seq, mask, sc)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        # validate
        model.eval()
        val_loader = DataLoader(EOTDataset(val_recs), batch_size=64, shuffle=False)
        preds, trues = [], []
        with torch.no_grad():
            for seq, mask, sc, labels in val_loader:
                logits = model(seq, mask, sc)
                preds.extend(torch.sigmoid(logits).numpy())
                trues.extend(labels.numpy())
        val_auc = auc_score(trues, preds)
        pbar.set_postfix({"loss": f"{epoch_loss/max(1,len(train_loader)):.3f}",
                          "val_auc": f"{val_auc:.3f}", "best": f"{best_auc:.3f}"})
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_auc


def predict(model, records):
    model.eval()
    loader = DataLoader(EOTDataset(records), batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for seq, mask, sc, _ in loader:
            logits = model(seq, mask, sc)
            preds.extend(torch.sigmoid(logits).numpy())
    return np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--out_en", default="predictions_en.csv")
    ap.add_argument("--out_hi", default="predictions_hi.csv")
    ap.add_argument("--model_out", default="model_lstm.pt")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    all_records, all_groups, all_keys = [], [], []
    per_dir = {}

    for d in args.data_dirs:
        print(f"Loading {d} ...")
        records, groups, keys = load_dataset(d)
        per_dir[d] = (records, groups, keys)
        all_records.extend(records)
        all_groups.extend(groups)
        all_keys.extend([(k[0], k[1], k[2], d) for k in keys])

    print(f"Total: {len(all_records)} pauses\n")

    # 5-fold OOF
    gkf = GroupKFold(n_splits=5)
    indices = np.arange(len(all_records))
    oof = np.zeros(len(all_records))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(indices, indices, all_groups)):
        t0 = time.time()
        train_recs = [all_records[i] for i in tr_idx]
        val_recs = [all_records[i] for i in va_idx]
        model, best_auc = train_one_fold(train_recs, val_recs, epochs=args.epochs)
        oof[va_idx] = predict(model, val_recs)
        print(f"  Fold {fold+1}: best val AUC = {best_auc:.3f}  ({time.time()-t0:.1f}s)")

    cv_auc = auc_score([r[3] for r in all_records], oof)
    print(f"\n5-fold CV AUC (LSTM): {cv_auc:.3f}")

    # Final model on ALL data
    print("\nTraining final model on all data ...")
    t0 = time.time()
    final_model, _ = train_one_fold(all_records, all_records, epochs=args.epochs)
    torch.save({
        "model_state": final_model.state_dict(),
        "n_frame_feats": N_FRAME_FEATS,
        "n_scalar_feats": N_SCALAR,
    }, args.model_out)
    print(f"Saved {args.model_out}  ({time.time()-t0:.1f}s)")

    # Write OOF predictions per directory
    dir_lookup = {i: all_keys[i][3] for i in range(len(all_keys))}

    def write_preds(data_dir, out_path):
        idxs = [i for i, k in enumerate(all_keys) if k[3] == data_dir]
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["turn_id", "pause_index", "p_eot"])
            for i in idxs:
                tid, pi, _, _ = all_keys[i]
                w.writerow([tid, pi, f"{oof[i]:.4f}"])
        print(f"Wrote {len(idxs)} predictions -> {out_path}")

    for d in args.data_dirs:
        out = args.out_en if "english" in d else args.out_hi
        write_preds(d, out)


if __name__ == "__main__":
    main()
