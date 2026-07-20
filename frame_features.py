"""
Frame-level feature extraction for the LSTM model.
All features are causal: only audio from [0, pause_start) is used.

For each pause, we return a fixed-length sequence of T frames,
each frame has D features extracted at 10ms hop rate.

Frame features (per frame):
  0: log RMS energy (normalized per-turn)
  1: F0 in Hz — 0 if unvoiced (normalized per-turn)
  2: voiced flag (0/1)
  3: spectral centroid (normalized per-turn)
  4: spectral rolloff (normalized per-turn)
  5: ZCR
  6–18: 13 MFCCs (normalized per-turn mean/std)
"""
import numpy as np
import librosa
import soundfile as sf

SR = 16000        # all audio is 16 kHz mono
HOP = 160         # 10 ms hop at 16 kHz
N_FFT = 512       # 32 ms window
N_MFCC = 13
SEQ_LEN = 150     # 1.5 seconds of context (150 * 10ms)
N_FRAME_FEATS = 6 + N_MFCC  # 19 features per frame


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x, SR


def _safe_norm(arr, mean=None, std=None, eps=1e-8):
    if mean is None:
        mean = arr.mean()
    if std is None:
        std = arr.std() + eps
    return (arr - mean) / std


def extract_frame_sequence(x, sr, pause_start, speaker_stats=None):
    """
    Returns (seq, mask) where:
      seq  : (SEQ_LEN, N_FRAME_FEATS) float32
      mask : (SEQ_LEN,) bool — True where real frames exist

    speaker_stats: dict with per-turn normalization stats (computed from
                   audio before pause_start). If None, computed here.
    """
    end_sample = int(pause_start * sr)
    start_sample = max(0, end_sample - SEQ_LEN * HOP)
    seg = x[start_sample:end_sample]

    seq = np.zeros((SEQ_LEN, N_FRAME_FEATS), dtype=np.float32)
    mask = np.zeros(SEQ_LEN, dtype=bool)

    if len(seg) < N_FFT:
        return seq, mask

    # ---- log RMS energy ----
    rms = librosa.feature.rms(y=seg, frame_length=N_FFT, hop_length=HOP)[0]
    log_rms = np.log(rms + 1e-8)

    # ---- MFCC ----
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP)  # (N_MFCC, T)

    # ---- spectral centroid & rolloff ----
    centroid = librosa.feature.spectral_centroid(y=seg, sr=sr,
                                                  n_fft=N_FFT, hop_length=HOP)[0]
    rolloff = librosa.feature.spectral_rolloff(y=seg, sr=sr,
                                                n_fft=N_FFT, hop_length=HOP)[0]

    # ---- ZCR ----
    zcr = librosa.feature.zero_crossing_rate(seg, frame_length=N_FFT,
                                              hop_length=HOP)[0]

    # ---- F0 via pyin (probabilistic, more robust than autocorr) ----
    try:
        f0, voiced_flag, _ = librosa.pyin(
            seg, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr, hop_length=HOP, fill_na=0.0,
        )
        voiced_flag = voiced_flag.astype(np.float32)
    except Exception:
        f0 = np.zeros_like(log_rms)
        voiced_flag = np.zeros_like(log_rms)

    T = min(len(log_rms), len(f0), len(centroid), len(rolloff), len(zcr),
            mfcc.shape[1])

    # ---- per-turn speaker normalization ----
    if speaker_stats is None:
        # compute from this segment
        f0_voiced = f0[:T][voiced_flag[:T] > 0.5]
        speaker_stats = {
            "rms_mean": log_rms[:T].mean(),
            "rms_std": log_rms[:T].std() + 1e-8,
            "f0_mean": f0_voiced.mean() if len(f0_voiced) > 0 else 200.0,
            "f0_std": f0_voiced.std() + 1e-8 if len(f0_voiced) > 0 else 50.0,
            "cent_mean": centroid[:T].mean(),
            "cent_std": centroid[:T].std() + 1e-8,
            "roll_mean": rolloff[:T].mean(),
            "roll_std": rolloff[:T].std() + 1e-8,
            "mfcc_mean": mfcc[:, :T].mean(axis=1),
            "mfcc_std": mfcc[:, :T].std(axis=1) + 1e-8,
        }

    # ---- assemble frame matrix (most recent SEQ_LEN frames) ----
    n_real = min(T, SEQ_LEN)
    offset = SEQ_LEN - n_real  # pad at the start (older = left)

    log_rms_n = (log_rms[:T] - speaker_stats["rms_mean"]) / speaker_stats["rms_std"]
    f0_n = np.where(
        voiced_flag[:T] > 0.5,
        (f0[:T] - speaker_stats["f0_mean"]) / speaker_stats["f0_std"],
        0.0,
    )
    cent_n = (centroid[:T] - speaker_stats["cent_mean"]) / speaker_stats["cent_std"]
    roll_n = (rolloff[:T] - speaker_stats["roll_mean"]) / speaker_stats["roll_std"]
    mfcc_n = ((mfcc[:, :T] - speaker_stats["mfcc_mean"][:, None])
              / speaker_stats["mfcc_std"][:, None])  # (N_MFCC, T)

    src_start = max(0, T - SEQ_LEN)
    src_end = T
    dst_start = offset
    dst_end = SEQ_LEN

    seq[dst_start:dst_end, 0] = log_rms_n[src_start:src_end]
    seq[dst_start:dst_end, 1] = f0_n[src_start:src_end]
    seq[dst_start:dst_end, 2] = voiced_flag[src_start:src_end]
    seq[dst_start:dst_end, 3] = cent_n[src_start:src_end]
    seq[dst_start:dst_end, 4] = roll_n[src_start:src_end]
    seq[dst_start:dst_end, 5] = zcr[src_start:src_end]
    seq[dst_start:dst_end, 6:] = mfcc_n[:, src_start:src_end].T

    mask[dst_start:dst_end] = True

    return seq, mask, speaker_stats
