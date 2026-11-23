"""Sintesis nada acuan lokal untuk latihan bernyanyi."""

import io
import numpy as np
import soundfile as sf

from src.config import INTERNAL_SAMPLE_RATE


def synthesize_reference_tone_wav(
    frequency_hz: float,
    duration_sec: float = 2.5,
    sample_rate: int = INTERNAL_SAMPLE_RATE,
    amplitude: float = 0.35,
) -> bytes:
    """Buat byte WAV nada acuan lokal dengan fade-in dan fade-out lembut."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0.0, duration_sec, num_samples, endpoint=False)
    wave = amplitude * np.sin(2.0 * np.pi * frequency_hz * t)

    fade_samples = min(int(0.05 * sample_rate), num_samples // 4)
    if fade_samples > 0:
        wave[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)
        wave[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)

    buffer = io.BytesIO()
    sf.write(buffer, wave.astype(np.float32), sample_rate, format="WAV")
    return buffer.getvalue()
