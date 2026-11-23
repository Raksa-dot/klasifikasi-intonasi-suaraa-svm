"""Estimasi pitch/F0 menggunakan librosa.pyin beserta penyaringan outlier dan voiced mask."""

from dataclasses import dataclass
import librosa
import numpy as np

from src.config import (
    EXTRACTOR_VERSION,
    FRAME_LENGTH,
    HOP_LENGTH,
    INTERNAL_SAMPLE_RATE,
    SINGING_FMAX_HZ,
    SINGING_FMIN_HZ,
    SPEECH_FMAX_HZ,
    SPEECH_FMIN_HZ,
)


@dataclass
class PitchContour:
    """Hasil ekstraksi F0 per frame beserta konfigurasi ekstraksinya."""

    timestamps: np.ndarray
    f0_hz: np.ndarray
    voiced_mask: np.ndarray
    voiced_prob: np.ndarray
    sample_rate: int
    fmin_hz: float
    fmax_hz: float
    frame_length: int
    hop_length: int
    time_offset_sec: float = 0.0
    extractor_version: str = EXTRACTOR_VERSION


def _filter_isolated_octave_outliers(f0_hz: np.ndarray, max_jump_semitones: float = 10.5) -> np.ndarray:
    """Saring loncatan outlier 1-frame tunggal (>10.5 semitone dari kedua tetangganya) tanpa mengubah vibrato."""
    cleaned = f0_hz.copy()
    n = cleaned.size
    for i in range(1, n - 1):
        curr = cleaned[i]
        prev_v = cleaned[i - 1]
        next_v = cleaned[i + 1]
        if np.isfinite(curr) and np.isfinite(prev_v) and np.isfinite(next_v) and curr > 0 and prev_v > 0 and next_v > 0:
            jump_prev = abs(12.0 * np.log2(curr / prev_v))
            jump_next = abs(12.0 * np.log2(curr / next_v))
            neighbor_diff = abs(12.0 * np.log2(next_v / prev_v))
            if jump_prev > max_jump_semitones and jump_next > max_jump_semitones and neighbor_diff < 4.0:
                cleaned[i] = 0.5 * (prev_v + next_v)
    return cleaned


def extract_pitch_contour(
    mono_samples: np.ndarray,
    sample_rate: int = INTERNAL_SAMPLE_RATE,
    mode: str = "SPEAKING",
    time_offset_sec: float = 0.0,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> PitchContour:
    """Ekstraksi kontur F0 menggunakan librosa.pyin sesuai mode latihan."""
    is_singing = mode.upper() == "SINGING"
    fmin = SINGING_FMIN_HZ if is_singing else SPEECH_FMIN_HZ
    fmax = SINGING_FMAX_HZ if is_singing else SPEECH_FMAX_HZ

    if mono_samples is None or mono_samples.size < frame_length:
        empty = np.array([], dtype=float)
        empty_bool = np.array([], dtype=bool)
        return PitchContour(
            timestamps=empty,
            f0_hz=empty,
            voiced_mask=empty_bool,
            voiced_prob=empty,
            sample_rate=sample_rate,
            fmin_hz=fmin,
            fmax_hz=fmax,
            frame_length=frame_length,
            hop_length=hop_length,
            time_offset_sec=time_offset_sec,
        )

    f0, voiced_flag, voiced_probs = librosa.pyin(
        mono_samples,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    timestamps = librosa.times_like(f0, sr=sample_rate, hop_length=hop_length) + float(time_offset_sec)
    valid_mask = np.logical_and(voiced_flag, np.isfinite(f0)) & (f0 > 0)
    clean_f0 = np.where(valid_mask, f0, np.nan)
    clean_f0 = _filter_isolated_octave_outliers(clean_f0)
    valid_mask = np.isfinite(clean_f0) & (clean_f0 > 0)

    return PitchContour(
        timestamps=timestamps,
        f0_hz=clean_f0,
        voiced_mask=valid_mask,
        voiced_prob=np.asarray(voiced_probs, dtype=float),
        sample_rate=sample_rate,
        fmin_hz=fmin,
        fmax_hz=fmax,
        frame_length=frame_length,
        hop_length=hop_length,
        time_offset_sec=time_offset_sec,
    )
