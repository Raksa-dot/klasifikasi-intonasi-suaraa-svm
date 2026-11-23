"""Validasi input audio dan pemeriksaan kualitas sinyal sebelum analisis pitch."""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from src.config import (
    CLIPPING_THRESHOLD,
    MAX_CLIPPING_RATIO,
    MAX_DURATION_SEC,
    MAX_UPLOAD_SIZE_MB,
    MIN_DURATION_SEC,
    MIN_RMS_ENERGY,
)


@dataclass
class AudioValidationResult:
    """Hasil pemeriksaan validitas dan kualitas sinyal awal."""

    is_valid: bool
    duration_sec: float
    sample_rate: int
    rms_energy: float
    peak_amplitude: float
    clipping_ratio: float
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_raw_signal(
    samples: np.ndarray,
    sample_rate: int,
    file_size_bytes: Optional[int] = None,
    min_duration_sec: float = MIN_DURATION_SEC,
    max_duration_sec: float = MAX_DURATION_SEC,
) -> AudioValidationResult:
    """Ukur ukuran, durasi, energi RMS, dan clipping dari sinyal asli sebelum normalisasi."""
    reasons: List[str] = []
    warnings: List[str] = []

    if file_size_bytes is not None:
        max_bytes = int(MAX_UPLOAD_SIZE_MB * 1024 * 1024)
        if file_size_bytes > max_bytes:
            reasons.append(
                f"Ukuran file ({file_size_bytes / (1024**2):.2f} MB) melebihi batas maksimum {MAX_UPLOAD_SIZE_MB} MB."
            )

    if samples is None or samples.size == 0 or sample_rate <= 0:
        return AudioValidationResult(
            is_valid=False,
            duration_sec=0.0,
            sample_rate=max(0, int(sample_rate or 0)),
            rms_energy=0.0,
            peak_amplitude=0.0,
            clipping_ratio=0.0,
            reasons=reasons + ["Audio kosong atau tidak dapat didekode."],
        )

    mono = np.mean(samples, axis=1) if samples.ndim > 1 else samples.astype(np.float32)
    if not np.all(np.isfinite(mono)):
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

    duration_sec = float(len(mono) / sample_rate)
    rms_energy = float(np.sqrt(np.mean(np.square(mono))))
    peak_amp = float(np.max(np.abs(mono))) if mono.size > 0 else 0.0
    clipping_ratio = float(np.mean(np.abs(mono) >= CLIPPING_THRESHOLD))

    if duration_sec < min_duration_sec:
        reasons.append(
            f"Durasi rekaman ({duration_sec:.2f} detik) terlalu pendek. Minimal {min_duration_sec:.1f} detik."
        )
    if duration_sec > max_duration_sec:
        reasons.append(
            f"Durasi rekaman ({duration_sec:.2f} detik) melebihi batas satu latihan pendek ({max_duration_sec:.1f} detik)."
        )
    if rms_energy < MIN_RMS_ENERGY:
        reasons.append(
            "Sinyal suara terlalu pelan atau hening. Silakan merekam ulang lebih dekat ke mikrofon di tempat tenang."
        )
    if clipping_ratio > MAX_CLIPPING_RATIO:
        reasons.append(
            f"Terdeteksi clipping berat ({clipping_ratio * 100:.1f}% sampel jenuh). Kurangi volume mikrofon atau jaga jarak."
        )
    elif clipping_ratio > 0.005:
        warnings.append(
            f"Terdapat sedikit puncak sinyal mendekati batas saturasi ({clipping_ratio * 100:.2f}% sampel)."
        )

    return AudioValidationResult(
        is_valid=len(reasons) == 0,
        duration_sec=duration_sec,
        sample_rate=int(sample_rate),
        rms_energy=rms_energy,
        peak_amplitude=peak_amp,
        clipping_ratio=clipping_ratio,
        reasons=reasons,
        warnings=warnings,
    )
