"""
Feature extraction for End-of-Turn detection.
All features are causal: use only audio from [0, pause_start).

Key design decisions after error analysis:
- Focus on last 0.5s (not 1.5s) for prosodic slope features — the
  end-of-utterance signal is local, averaging 1.5s dilutes it.
- Speaker-normalized F0 using per-turn running stats.
- Language flag as explicit feature (Hindi model needs different weighting).
- Pause position features: first-pause EOTs were systematically missed.
"""
import numpy as np
import soundfile as sf
import librosa

SR = 16000
HOP = 160       # 10ms at 16kHz
N_FFT = 512


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def _seg(x, sr, pause_start, window_s):
    end = int(pause_start * sr)
    start = max(0, end - int(window_s * sr))
    return x[start:end]


def _frame_energy_db(seg, sr):
    fl = int(sr * 25 / 1000)
    hp = int(sr * 10 / 1000)
    if len(seg) < fl:
        return np.array([-60.0])
    n = 1 + (len(seg) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(seg[idx] ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def _slope(arr):
    if len(arr) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])


def extract_features(x, sr, pause_start, pause_index, prior_pauses,
                     turn_f0_stats=None, lang="en"):
    """
    Extract 60 causal features for one pause.

    Parameters
    ----------
    x, sr         : audio
    pause_start   : float (seconds) — start of this pause
    pause_index   : int — 0-based index of this pause within the turn
    prior_pauses  : list of pause_start values seen so far in this turn
    turn_f0_stats : dict with 'mean', 'std' for speaker F0 normalization
                    (computed from prior pauses; None on first pause)
    lang          : "en" or "hi"
    """
    MIN = sr // 20   # 50ms minimum segment

    # ---- segments at multiple windows ----
    seg05 = _seg(x, sr, pause_start, 0.5)   # last 0.5s — most informative
    seg10 = _seg(x, sr, pause_start, 1.0)   # last 1.0s
    seg15 = _seg(x, sr, pause_start, 1.5)   # last 1.5s

    if len(seg05) < MIN:
        return np.zeros(60, dtype=np.float32)

    # ================================================================
    # 1. ENERGY FEATURES (focus on last 0.5s for slope, not 1.5s)
    # ================================================================
    e05 = _frame_energy_db(seg05, sr)
    e10 = _frame_energy_db(seg10, sr) if len(seg10) >= MIN else e05
    e15 = _frame_energy_db(seg15, sr) if len(seg15) >= MIN else e10

    energy_slope_05 = _slope(e05)           # most local — key EOT signal
    energy_slope_10 = _slope(e10)
    energy_final_05 = e05[-3:].mean()
    energy_mean_05 = e05.mean()
    energy_decay = energy_final_05 - e05[:max(1, len(e05)//2)].mean()  # neg = falling
    energy_std_05 = e05.std()

    # last 0.3s vs first 0.2s of the 0.5s window
    split = max(1, len(e05) * 2 // 5)
    energy_end_vs_start = e05[split:].mean() - e05[:split].mean()

    # ================================================================
    # 2. F0 FEATURES — speaker-normalized, last 0.5s focus
    # ================================================================
    def get_f0(seg):
        if len(seg) < N_FFT:
            return np.array([0.0]), np.array([False])
        try:
            f0, vf, _ = librosa.pyin(seg, fmin=50, fmax=500, sr=sr,
                                      hop_length=HOP, fill_na=0.0)
            return f0, vf.astype(bool)
        except Exception:
            return np.zeros(len(seg) // HOP), np.zeros(len(seg) // HOP, dtype=bool)

    f0_05, vf_05 = get_f0(seg05)
    f0_10, vf_10 = get_f0(seg10) if len(seg10) >= MIN else (f0_05, vf_05)

    voiced_05 = f0_05[vf_05]
    voiced_10 = f0_10[vf_10]

    voiced_frac_05 = vf_05.mean() if len(vf_05) > 0 else 0.0
    voiced_frac_10 = vf_10.mean() if len(vf_10) > 0 else 0.0

    # speaker normalization
    if turn_f0_stats and len(voiced_10) == 0:
        spk_f0_mean = turn_f0_stats["mean"]
        spk_f0_std = turn_f0_stats["std"]
    elif len(voiced_10) > 0:
        spk_f0_mean = voiced_10.mean()
        spk_f0_std = voiced_10.std() + 1e-8
    else:
        spk_f0_mean = 150.0
        spk_f0_std = 30.0

    if len(voiced_05) >= 3:
        f0_final_norm = (voiced_05[-3:].mean() - spk_f0_mean) / spk_f0_std
        f0_mean_norm = (voiced_05.mean() - spk_f0_mean) / spk_f0_std
        f0_final_ratio = voiced_05[-3:].mean() / (voiced_05.mean() + 1e-8)
        f0_std_05 = voiced_05.std()

        # slope over last 8 voiced frames
        vpos = np.where(vf_05)[0]
        tail = min(8, len(vpos))
        f0_slope_05 = float(np.polyfit(vpos[-tail:], voiced_05[-tail:], 1)[0])

        # last voiced region duration (syllable lengthening)
        last_uv = len(vf_05) - 1
        while last_uv >= 0 and vf_05[last_uv]:
            last_uv -= 1
        last_voiced_dur = (len(vf_05) - 1 - last_uv) * (HOP / sr)
    else:
        f0_final_norm = 0.0
        f0_mean_norm = 0.0
        f0_final_ratio = 1.0
        f0_std_05 = 0.0
        f0_slope_05 = 0.0
        last_voiced_dur = 0.0

    if len(voiced_10) >= 5:
        vpos10 = np.where(vf_10)[0]
        tail10 = min(12, len(vpos10))
        f0_slope_10 = float(np.polyfit(vpos10[-tail10:], voiced_10[-tail10:], 1)[0])
        f0_slope_10_norm = f0_slope_10 / (spk_f0_std + 1e-8)
    else:
        f0_slope_10 = 0.0
        f0_slope_10_norm = 0.0

    # ================================================================
    # 3. SPECTRAL FEATURES (last 0.5s)
    # ================================================================
    if len(seg05) >= N_FFT:
        centroid = librosa.feature.spectral_centroid(
            y=seg05, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
        rolloff = librosa.feature.spectral_rolloff(
            y=seg05, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
        S = np.abs(librosa.stft(seg05, n_fft=N_FFT, hop_length=HOP))
        flux = np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0)) if S.shape[1] > 1 else np.array([0.0])
        zcr = librosa.feature.zero_crossing_rate(seg05, frame_length=N_FFT, hop_length=HOP)[0]

        centroid_slope = _slope(centroid)
        centroid_final = centroid[-3:].mean() if len(centroid) >= 3 else centroid.mean()
        rolloff_slope = _slope(rolloff)
        rolloff_final = rolloff[-3:].mean() if len(rolloff) >= 3 else rolloff.mean()
        flux_mean = flux.mean()
        zcr_mean = zcr.mean()
        zcr_final = zcr[-3:].mean() if len(zcr) >= 3 else zcr.mean()
    else:
        centroid_slope = centroid_final = 0.0
        rolloff_slope = rolloff_final = 0.0
        flux_mean = zcr_mean = zcr_final = 0.0

    # ================================================================
    # 4. MFCC FEATURES (last 0.5s — dynamic features are key)
    # ================================================================
    if len(seg05) >= N_FFT:
        mfcc = librosa.feature.mfcc(y=seg05, sr=sr, n_mfcc=13,
                                     n_fft=N_FFT, hop_length=HOP)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = mfcc_delta.mean(axis=1)
    else:
        mfcc_mean = np.zeros(13)
        mfcc_delta_mean = np.zeros(13)

    # ================================================================
    # 5. TURN-LEVEL CONTEXT FEATURES
    # ================================================================
    time_since_last = pause_start - prior_pauses[-1] if prior_pauses else pause_start
    n_prior = len(prior_pauses)

    # avg pause interval so far (rhythm of the turn)
    if len(prior_pauses) >= 2:
        intervals = np.diff(prior_pauses)
        avg_interval = intervals.mean()
    else:
        avg_interval = pause_start

    # speaking rate: voiced fraction over entire turn up to pause_start
    full_seg = x[:int(pause_start * sr)]
    if len(full_seg) > sr // 2:
        f0_full, vf_full = get_f0(full_seg[-int(sr * 3):])  # last 3s max
        turn_voiced_frac = vf_full.mean() if len(vf_full) > 0 else voiced_frac_10
    else:
        turn_voiced_frac = voiced_frac_10

    lang_flag = 1.0 if lang == "hi" else 0.0

    # ================================================================
    # ASSEMBLE
    # ================================================================
    base = np.array([
        # energy (0-6)
        energy_slope_05, energy_slope_10, energy_final_05,
        energy_decay, energy_end_vs_start, energy_std_05, energy_mean_05,
        # F0 (7-14)
        f0_slope_05, f0_slope_10, f0_slope_10_norm,
        f0_final_norm, f0_mean_norm, f0_final_ratio,
        f0_std_05, last_voiced_dur,
        # voiced (15-16)
        voiced_frac_05, voiced_frac_10,
        # spectral (17-23)
        centroid_slope, centroid_final,
        rolloff_slope, rolloff_final,
        flux_mean, zcr_mean, zcr_final,
        # turn context (24-30)
        float(pause_index), pause_start, float(n_prior),
        time_since_last, avg_interval, turn_voiced_frac,
        lang_flag,
    ], dtype=np.float32)

    feat = np.concatenate([base, mfcc_mean.astype(np.float32),
                           mfcc_delta_mean.astype(np.float32)])
    return feat  # shape (31 + 13 + 13,) = (57,)


FEATURE_NAMES = (
    ["e_slope_05", "e_slope_10", "e_final_05", "e_decay", "e_end_vs_start",
     "e_std_05", "e_mean_05",
     "f0_slope_05", "f0_slope_10", "f0_slope_10_norm",
     "f0_final_norm", "f0_mean_norm", "f0_final_ratio",
     "f0_std_05", "last_voiced_dur",
     "voiced_frac_05", "voiced_frac_10",
     "centroid_slope", "centroid_final",
     "rolloff_slope", "rolloff_final",
     "flux_mean", "zcr_mean", "zcr_final",
     "pause_index", "pause_start", "n_prior",
     "time_since_last", "avg_interval", "turn_voiced_frac", "lang_flag"]
    + [f"mfcc_{i}" for i in range(13)]
    + [f"mfcc_delta_{i}" for i in range(13)]
)
