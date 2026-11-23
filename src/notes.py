"""Fungsi konversi frekuensi (Hz), nomor MIDI, notasi nada musik, dan selisih cents."""

import math
from typing import Optional

from src.config import A4_MIDI_NUMBER, A4_REFERENCE_HZ

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def hz_to_midi(frequency_hz: float, a4_hz: float = A4_REFERENCE_HZ) -> Optional[float]:
    """Konversi frekuensi (Hz) ke nilai MIDI kontinu. Mengembalikan None jika <= 0 atau NaN."""
    if frequency_hz is None or math.isnan(frequency_hz) or frequency_hz <= 0:
        return None
    return A4_MIDI_NUMBER + 12.0 * math.log2(frequency_hz / a4_hz)


def midi_to_hz(midi_value: float, a4_hz: float = A4_REFERENCE_HZ) -> float:
    """Konversi nomor MIDI ke frekuensi (Hz)."""
    return a4_hz * (2.0 ** ((midi_value - A4_MIDI_NUMBER) / 12.0))


def hz_to_note_name(frequency_hz: float, a4_hz: float = A4_REFERENCE_HZ) -> Optional[str]:
    """Bulatkan frekuensi ke semitone terdekat untuk nama nada (mis. A2, C3, D4)."""
    midi_val = hz_to_midi(frequency_hz, a4_hz=a4_hz)
    if midi_val is None:
        return None
    nearest_midi = int(round(midi_val))
    note_index = nearest_midi % 12
    octave = (nearest_midi // 12) - 1
    return f"{NOTE_NAMES[note_index]}{octave}"


def calculate_cents_error(f0_hz: float, target_hz: float) -> Optional[float]:
    """Hitung selisih bertanda dalam cents terhadap frekuensi nada target.

    Nilai positif: lebih tinggi dari target; negatif: lebih rendah; nol: tepat pada target.
    """
    if (
        f0_hz is None
        or target_hz is None
        or math.isnan(f0_hz)
        or math.isnan(target_hz)
        or f0_hz <= 0
        or target_hz <= 0
    ):
        return None
    return 1200.0 * math.log2(f0_hz / target_hz)
