"""Unit test untuk validasi sinyal audio (hening, terlalu pendek, clipping, kosong, dan sinyal valid)."""

import numpy as np

from src.validation import validate_raw_signal


def test_reject_empty_audio() -> None:
    result = validate_raw_signal(np.array([], dtype=np.float32), 22050)
    assert not result.is_valid
    assert any("kosong" in r.lower() for r in result.reasons)


def test_reject_silent_audio() -> None:
    silent = np.zeros(22050 * 2, dtype=np.float32)
    result = validate_raw_signal(silent, 22050)
    assert not result.is_valid
    assert any("hening" in r.lower() or "pelan" in r.lower() for r in result.reasons)


def test_reject_too_short_audio() -> None:
    short_tone = 0.2 * np.sin(2.0 * np.pi * 220.0 * np.linspace(0, 0.2, int(22050 * 0.2)))
    result = validate_raw_signal(short_tone.astype(np.float32), 22050)
    assert not result.is_valid
    assert any("pendek" in r.lower() for r in result.reasons)


def test_reject_heavy_clipping_audio() -> None:
    t = np.linspace(0, 2.0, 22050 * 2)
    clipped = np.clip(5.0 * np.sin(2.0 * np.pi * 220.0 * t), -1.0, 1.0).astype(np.float32)
    result = validate_raw_signal(clipped, 22050)
    assert not result.is_valid
    assert any("clipping" in r.lower() for r in result.reasons)


def test_accept_clean_sine_audio() -> None:
    t = np.linspace(0, 2.0, 22050 * 2, endpoint=False)
    clean = (0.3 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    result = validate_raw_signal(clean, 22050)
    assert result.is_valid
    assert len(result.reasons) == 0
