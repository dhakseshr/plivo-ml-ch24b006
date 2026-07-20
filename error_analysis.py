"""
Error analysis: find worst-performing pauses in OOF predictions.
Prints the cases where the model is most wrong, grouped by error type.
Run: python error_analysis.py
"""
import csv
import os
import numpy as np

DATA_DIRS = [
    "eot_handout/eot_data/english",
    "eot_handout/eot_data/hindi",
]
PRED_FILES = ["predictions_en.csv", "predictions_hi.csv"]


def load(data_dir, pred_file):
    preds = {}
    with open(pred_file) as f:
        for r in csv.DictReader(f):
            preds[(r["turn_id"], int(r["pause_index"]))] = float(r["p_eot"])

    rows = []
    with open(os.path.join(data_dir, "labels.csv")) as f:
        for r in csv.DictReader(f):
            key = (r["turn_id"], int(r["pause_index"]))
            rows.append({
                "turn_id": r["turn_id"],
                "pause_index": int(r["pause_index"]),
                "pause_start": float(r["pause_start"]),
                "pause_dur": float(r["pause_end"]) - float(r["pause_start"]),
                "label": r["label"],
                "p_eot": preds[key],
                "audio_file": r["audio_file"],
                "lang": "en" if "english" in data_dir else "hi",
            })
    return rows


all_rows = []
for d, p in zip(DATA_DIRS, PRED_FILES):
    all_rows.extend(load(d, p))

y = np.array([1 if r["label"] == "eot" else 0 for r in all_rows])
p = np.array([r["p_eot"] for r in all_rows])

# AUC
order = np.argsort(p)
ranks = np.empty_like(order, dtype=float)
ranks[order] = np.arange(1, len(p) + 1)
n1, n0 = y.sum(), len(y) - y.sum()
auc = (ranks[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)
print(f"Overall OOF AUC: {auc:.3f}  ({n1} EOT, {n0} HOLD)\n")

# ---- Worst errors ----
errors = [(abs(r["p_eot"] - y_i), r, y_i) for r, y_i in zip(all_rows, y)]
errors.sort(key=lambda x: x[0], reverse=True)

print("=" * 70)
print("TOP 20 WORST ERRORS")
print("=" * 70)
print(f"{'lang':4} {'turn_id':12} {'pi':3} {'label':5} {'p_eot':6} {'error':6} "
      f"{'pause_s':8} {'pause_dur':9}")
for err, r, yi in errors[:20]:
    print(f"{r['lang']:4} {r['turn_id']:12} {r['pause_index']:3} "
          f"{r['label']:5} {r['p_eot']:6.3f} {err:6.3f} "
          f"{r['pause_start']:8.2f}s {r['pause_dur']:8.2f}s")

# ---- False positives (HOLD predicted as EOT) ----
hold_rows = [r for r, yi in zip(all_rows, y) if yi == 0]
hold_rows.sort(key=lambda r: r["p_eot"], reverse=True)
print(f"\n{'='*70}")
print("TOP 15 FALSE POSITIVES (HOLD predicted as EOT — cause interruptions)")
print(f"{'='*70}")
print(f"{'lang':4} {'turn_id':12} {'pi':3} {'p_eot':6} {'pause_start':11} "
      f"{'pause_dur':9} {'risk@100ms':10}")
for r in hold_rows[:15]:
    risk = "DANGER" if r["pause_dur"] > 0.1 else "safe"
    print(f"{r['lang']:4} {r['turn_id']:12} {r['pause_index']:3} "
          f"{r['p_eot']:6.3f} {r['pause_start']:11.2f}s "
          f"{r['pause_dur']:9.2f}s {risk:10}")

# ---- False negatives (EOT predicted as HOLD) ----
eot_rows = [r for r, yi in zip(all_rows, y) if yi == 1]
eot_rows.sort(key=lambda r: r["p_eot"])
print(f"\n{'='*70}")
print("TOP 15 FALSE NEGATIVES (EOT predicted as HOLD — cause long delays)")
print(f"{'='*70}")
print(f"{'lang':4} {'turn_id':12} {'pi':3} {'p_eot':6} {'pause_start':11} "
      f"{'pause_dur':9}")
for r in eot_rows[:15]:
    print(f"{r['lang']:4} {r['turn_id']:12} {r['pause_index']:3} "
          f"{r['p_eot']:6.3f} {r['pause_start']:11.2f}s {r['pause_dur']:9.2f}s")

# ---- Stats by language ----
print(f"\n{'='*70}")
print("STATS BY LANGUAGE")
print(f"{'='*70}")
for lang in ["en", "hi"]:
    lr = [r for r in all_rows if r["lang"] == lang]
    lp = np.array([r["p_eot"] for r in lr])
    ly = np.array([1 if r["label"] == "eot" else 0 for r in lr])
    eot_p = lp[ly == 1]
    hold_p = lp[ly == 0]
    print(f"\n{lang.upper()}:")
    print(f"  EOT  p_eot: mean={eot_p.mean():.3f}  median={np.median(eot_p):.3f}  "
          f"min={eot_p.min():.3f}  max={eot_p.max():.3f}")
    print(f"  HOLD p_eot: mean={hold_p.mean():.3f}  median={np.median(hold_p):.3f}  "
          f"min={hold_p.min():.3f}  max={hold_p.max():.3f}")

# ---- Pause position analysis ----
print(f"\n{'='*70}")
print("EOT RATE BY PAUSE POSITION (pause_index)")
print(f"{'='*70}")
for pi in range(6):
    subset = [r for r in all_rows if r["pause_index"] == pi]
    if not subset:
        continue
    eot_rate = sum(1 for r in subset if r["label"] == "eot") / len(subset)
    avg_p = np.mean([r["p_eot"] for r in subset])
    print(f"  pause_index={pi}: n={len(subset):3d}  EOT rate={eot_rate:.2f}  "
          f"avg p_eot={avg_p:.3f}")
