"""Unit test untuk konversi frekuensi, MIDI, nama nada, dan perhitungan selisih cents (termasuk beda oktaf)."""

import pytest

from src.notes import calculate_cents_error, hz_to_midi, hz_to_note_name, midi_to_hz


def test_a4_reference_conversion() -> None:
    assert hz_to_midi(440.0) == pytest.approx(69.0)
    assert midi_to_hz(69.0) == pytest.approx(440.0)
    assert hz_to_note_name(440.0) == "A4"
    assert hz_to_note_name(261.625565) == "C4"
    assert hz_to_note_name(220.0) == "A3"
    assert calculate_cents_error(440.0, 440.0) == pytest.approx(0.0)


def test_octave_cents_difference_preserved() -> None:
    # Kesalahan satu oktaf tidak boleh dianggap 0 cents; harus tepat +1200 atau -1200 cents
    assert calculate_cents_error(440.0, 220.0) == pytest.approx(1200.0)
    assert calculate_cents_error(110.0, 220.0) == pytest.approx(-1200.0)


def test_signed_cents_direction() -> None:
    # Lebih tinggi = positif; lebih rendah = negatif
    assert calculate_cents_error(225.0, 220.0) > 0.0
    assert calculate_cents_error(215.0, 220.0) < 0.0


def test_invalid_frequencies_return_none() -> None:
    assert hz_to_midi(0.0) is None
    assert hz_to_midi(-100.0) is None
    assert hz_to_midi(float("inf")) is None
    assert hz_to_midi(float("-inf")) is None
    assert hz_to_note_name(0.0) is None
    assert hz_to_note_name(float("inf")) is None
    assert calculate_cents_error(0.0, 440.0) is None
    assert calculate_cents_error(440.0, 0.0) is None
    assert calculate_cents_error(float("inf"), 440.0) is None
    assert calculate_cents_error(440.0, float("inf")) is None
    assert calculate_cents_error(float("nan"), 440.0) is None
