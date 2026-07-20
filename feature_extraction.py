"""
Feature extraction for End-of-Turn detection.
All features are causal: use only audio from [0, pause_start).
"""
import numpy as np
import soundfile as sf
import librosa

FRAME_MS = 25
HOP_MS = 10


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def frames(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x, sr):
    fr = frames(x, sr)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def autocorr_f0(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.30):
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


def f0_contour(x, sr, frame_ms=40, hop_ms=HOP_MS):
    fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    return np.array([autocorr_f0(f, sr) for f in fr], dtype=np.float32)


def speech_before(x, sr, pause_start, window_s=1.5):
    end = int(pause_start * sr)
    start = max(0, end - int(window_s * sr))
    return x[start:end]


def last_voiced_duration(f0, hop_s):
    """Duration of the last continuous voiced region (syllable lengthening proxy)."""
    voiced_mask = f0 > 0
    if not voiced_mask.any():
        return 0.0
    idx = len(voiced_mask) - 1
    while idx >= 0 and voiced_mask[idx]:
        idx -= 1
    return (len(voiced_mask) - 1 - idx) * hop_s


def extract_features(x, sr, pause_start, pause_index, turn_pauses_so_far):
    """
    Extract causal prosodic + temporal features for one pause.

    Parameters
    ----------
    x : audio array
    sr : sample rate
    pause_start : float, seconds
    pause_index : int
    turn_pauses_so_far : list of (pause_start_s,) for prior pauses in this turn
    """
    hop_s = HOP_MS / 1000.0

    seg15 = speech_before(x, sr, pause_start, window_s=1.5)
    seg05 = speech_before(x, sr, pause_start, window_s=0.5)

    MIN_SAMPLES = sr // 10
    if len(seg15) < MIN_SAMPLES:
        return np.zeros(20, dtype=np.float32)

    # ---- energy features ----
    e15 = frame_energy_db(seg15, sr)
    e05 = frame_energy_db(seg05, sr) if len(seg05) >= MIN_SAMPLES else e15[-5:]

    energy_mean = e15.mean()
    energy_final = e15[-5:].mean()
    energy_ratio = energy_final - energy_mean          # negative = energy falling

    if len(e15) >= 5:
        xs = np.arange(len(e15))
        energy_slope = float(np.polyfit(xs, e15, 1)[0])
    else:
        energy_slope = 0.0

    # energy last 0.5s vs rest of 1.5s window
    energy_end_vs_start = e05.mean() - (e15[:-len(e05)].mean() if len(e15) > len(e05) else e15.mean())

    # ---- F0 features ----
    f0_15 = f0_contour(seg15, sr)
    voiced = f0_15[f0_15 > 0]
    voiced_frac = len(voiced) / (len(f0_15) + 1e-8)

    if len(voiced) >= 3:
        f0_final = voiced[-3:].mean()
        f0_mean = voiced.mean()
        f0_final_ratio = f0_final / (f0_mean + 1e-8)
        f0_std = voiced.std()
    else:
        f0_final = 0.0
        f0_mean = 0.0
        f0_final_ratio = 1.0
        f0_std = 0.0

    if len(voiced) >= 6:
        voiced_pos = np.where(f0_15 > 0)[0]
        tail = min(10, len(voiced_pos))
        f0_slope = float(np.polyfit(voiced_pos[-tail:], voiced[-tail:], 1)[0])
    else:
        f0_slope = 0.0

    last_vd = last_voiced_duration(f0_15, hop_s)

    # ---- full-turn speaking rate (voiced frac up to pause_start) ----
    full_seg = x[:int(pause_start * sr)]
    if len(full_seg) > sr // 2:
        f0_full = f0_contour(full_seg, sr)
        turn_voiced_frac = (f0_full > 0).mean()
    else:
        turn_voiced_frac = voiced_frac

    # ---- ZCR ----
    if len(seg05) > 1:
        zcr = float(((seg05[:-1] * seg05[1:]) < 0).mean())
    else:
        zcr = 0.0

    # ---- turn-level temporal context ----
    # pause_index: later pauses more likely EOT
    # pause_start: longer turns more likely complete
    # time since last pause
    if pause_index > 0 and turn_pauses_so_far:
        time_since_last = pause_start - turn_pauses_so_far[-1]
    else:
        time_since_last = pause_start  # first pause: time from turn start

    # ---- MFCC features (last 0.5s, language-agnostic spectral info) ----
    if len(seg05) >= MIN_SAMPLES:
        mfcc = librosa.feature.mfcc(y=seg05, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
        mfcc_mean = mfcc.mean(axis=1)           # shape (13,)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = mfcc_delta.mean(axis=1)  # shape (13,), captures dynamics
    else:
        mfcc_mean = np.zeros(13)
        mfcc_delta_mean = np.zeros(13)

    # ---- spectral features (centroid, rolloff, flux) ----
    if len(seg15) >= MIN_SAMPLES:
        hop = 160
        centroid = librosa.feature.spectral_centroid(y=seg15, sr=sr, n_fft=512, hop_length=hop)[0]
        rolloff = librosa.feature.spectral_rolloff(y=seg15, sr=sr, n_fft=512, hop_length=hop)[0]
        S = np.abs(librosa.stft(seg15, n_fft=512, hop_length=hop))
        flux = np.sqrt(np.mean(np.diff(S, axis=1)**2, axis=0))  # spectral flux per frame

        def feat_stat(arr):
            if len(arr) < 2:
                return 0.0, arr.mean() if len(arr) else 0.0, arr.mean() if len(arr) else 0.0
            slope = float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
            final = arr[-min(5, len(arr)):].mean()
            mean = arr.mean()
            return slope, final, mean

        centroid_slope, centroid_final, centroid_mean = feat_stat(centroid)
        rolloff_slope, rolloff_final, rolloff_mean = feat_stat(rolloff)
        flux_mean = flux.mean() if len(flux) else 0.0
        flux_final = flux[-5:].mean() if len(flux) >= 5 else flux_mean
    else:
        centroid_slope = centroid_final = centroid_mean = 0.0
        rolloff_slope = rolloff_final = rolloff_mean = 0.0
        flux_mean = flux_final = 0.0

    # ---- last 300ms energy slope (most local signal) ----
    seg03 = speech_before(x, sr, pause_start, window_s=0.3)
    if len(seg03) >= MIN_SAMPLES // 3:
        e03 = frame_energy_db(seg03, sr)
        if len(e03) >= 3:
            energy_slope_300ms = float(np.polyfit(np.arange(len(e03)), e03, 1)[0])
        else:
            energy_slope_300ms = energy_slope
    else:
        energy_slope_300ms = energy_slope

    base_features = np.array([
        energy_mean,           # 0
        energy_final,          # 1
        energy_ratio,          # 2  negative = falling energy
        energy_slope,          # 3  negative = falling energy
        energy_end_vs_start,   # 4
        energy_slope_300ms,    # 5  local energy slope
        f0_slope,              # 6  negative = falling pitch (EOT signal)
        f0_final,              # 7
        f0_final_ratio,        # 8  < 1 = pitch fell relative to mean
        f0_mean,               # 9
        f0_std,                # 10
        voiced_frac,           # 11
        last_vd,               # 12 syllable lengthening
        turn_voiced_frac,      # 13 speaking rate over whole turn
        zcr,                   # 14
        float(pause_index),    # 15 position in turn
        pause_start,           # 16 absolute time
        float(len(seg15)) / sr,  # 17 speech context length
        time_since_last,       # 18 inter-pause interval
        float(len(voiced)),    # 19 voiced frame count
        f0_final_ratio * (1.0 if f0_slope < 0 else 0.0),  # 20 falling F0 composite
        centroid_slope,        # 21 spectral centroid trend
        centroid_final,        # 22
        centroid_mean,         # 23
        rolloff_slope,         # 24
        rolloff_final,         # 25
        rolloff_mean,          # 26
        flux_mean,             # 27 spectral flux (high = dynamic = hold)
        flux_final,            # 28
    ], dtype=np.float32)

    features = np.concatenate([base_features, mfcc_mean.astype(np.float32),
                                mfcc_delta_mean.astype(np.float32)])
    return features


BASE_FEATURE_NAMES = [
    "energy_mean", "energy_final", "energy_ratio", "energy_slope",
    "energy_end_vs_start", "energy_slope_300ms",
    "f0_slope", "f0_final", "f0_final_ratio",
    "f0_mean", "f0_std", "voiced_frac", "last_voiced_dur",
    "turn_voiced_frac", "zcr", "pause_index", "pause_start",
    "speech_context_len", "time_since_last", "voiced_frame_count",
    "f0_fall_composite",
    "centroid_slope", "centroid_final", "centroid_mean",
    "rolloff_slope", "rolloff_final", "rolloff_mean",
    "flux_mean", "flux_final",
]
FEATURE_NAMES = (BASE_FEATURE_NAMES
                 + [f"mfcc_{i}" for i in range(13)]
                 + [f"mfcc_delta_{i}" for i in range(13)])
