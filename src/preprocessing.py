"""Preprocessing audio: konversi mono, penghilangan DC offset, resampling, dan pencatatan offset waktu."""

from dataclasses import dataclass
import librosa
import numpy as np

from src.config import INTERNAL_SAMPLE_RATE


@dataclass
class PreprocessedAudio:
    """Sinyal audio mono hasil preprocessing beserta informasi waktu aslinya."""

    samples: np.ndarray
    sample_rate: int
    original_sample_rate: int
    original_duration_sec: float
    processed_duration_sec: float
    time_offset_sec: float = 0.0


def preprocess_audio(
    samples: np.ndarray,
    orig_sr: int,
    target_sr: int = INTERNAL_SAMPLE_RATE,
    trim_silence: bool = False,
    top_db: float = 35.0,
) -> PreprocessedAudio:
    """Konversi sinyal ke mono, hilangkan DC offset, dan resample ke sample rate internal."""
    mono = np.mean(samples, axis=1) if samples.ndim > 1 else samples.astype(np.float32)
    mono = np.asarray(mono, dtype=np.float32)
    orig_duration = float(len(mono) / orig_sr) if orig_sr > 0 else 0.0

    # Hilangkan DC offset
    if mono.size > 0:
        mono = mono - float(np.mean(mono))

    if orig_sr != target_sr and mono.size > 0:
        mono = librosa.resample(mono, orig_sr=orig_sr, target_sr=target_sr)

    time_offset_sec = 0.0
    if trim_silence and mono.size > 0:
        trimmed, idx = librosa.effects.trim(mono, top_db=top_db)
        if trimmed.size >= int(0.4 * target_sr):
            mono = trimmed
            time_offset_sec = float(idx[0] / target_sr)

    proc_duration = float(len(mono) / target_sr) if target_sr > 0 else 0.0
    return PreprocessedAudio(
        samples=mono,
        sample_rate=target_sr,
        original_sample_rate=int(orig_sr),
        original_duration_sec=orig_duration,
        processed_duration_sec=proc_duration,
        time_offset_sec=time_offset_sec,
    )
