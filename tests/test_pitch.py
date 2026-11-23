"""Unit test untuk ekstraksi pitch, penanganan unvoiced NaN, dan pemisahan ketepatan vs kestabilan nada."""

import numpy as np
import pytest

from src.features import (
    compute_relative_semitones,
    extract_intonation_feature_dict,
    summarize_singing_pitch,
)
from src.pitch import PitchContour, extract_pitch_contour


def test_compute_relative_semitones_preserves_nan() -> None:
    f0 = np.array([np.nan, 220.0, 220.0, 440.0, np.nan], dtype=float)
    rel = compute_relative_semitones(f0)
    assert np.isnan(rel[0])
    assert np.isnan(rel[4])
    assert rel[1] == pytest.approx(0.0)
    assert rel[3] == pytest.approx(12.0)


def test_singing_stability_separated_from_target_accuracy() -> None:
    # Nada yang sangat stabil di 246.94 Hz (B3) saat target 220.0 Hz (A3):
    # Harus memiliki variasi kestabilan (MAD) = 0.0 cents, tetapi selisih terhadap target ~ +200 cents!
    n_frames = 40
    timestamps = np.arange(n_frames) * 0.02
    f0_hz = np.full(n_frames, 246.94165, dtype=float)
    voiced = np.ones(n_frames, dtype=bool)
    contour = PitchContour(
        timestamps=timestamps,
        f0_hz=f0_hz,
        voiced_mask=voiced,
        voiced_prob=np.ones(n_frames, dtype=float),
        sample_rate=22050,
        fmin_hz=65.0,
        fmax_hz=1000.0,
        frame_length=2048,
        hop_length=256,
    )
    summary = summarize_singing_pitch(contour, target_hz=220.0, tolerance_cents=50.0)
    assert summary is not None
    assert summary["stability_mad_cents"] == pytest.approx(0.0, abs=1e-4)
    assert summary["median_signed_cents_error"] == pytest.approx(200.0, abs=0.5)
    assert summary["in_tolerance_ratio"] == pytest.approx(0.0)


def test_extract_pitch_on_synthetic_sine() -> None:
    sr = 22050
    t = np.linspace(0.0, 1.5, int(sr * 1.5), endpoint=False)
    sine_220 = (0.4 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    contour = extract_pitch_contour(sine_220, sample_rate=sr, mode="SINGING")
    valid_f0 = contour.f0_hz[contour.voiced_mask]
    assert valid_f0.size >= 15
    assert float(np.median(valid_f0)) == pytest.approx(220.0, rel=0.03)
    feat = extract_intonation_feature_dict(contour)
    assert feat is not None
    assert len(feat) == 14
