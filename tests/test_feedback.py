"""Unit test untuk generator saran latihan berbicara dan bernyanyi."""

from src.feedback import generate_singing_feedback, generate_speaking_feedback


def test_speaking_feedback_when_summary_missing() -> None:
    msgs = generate_speaking_feedback(None, target_label="RISING")
    assert len(msgs) == 1
    assert "rekam ulang" in msgs[0].lower()


def test_singing_feedback_detects_octave_error() -> None:
    summary = {
        "median_signed_cents_error": -1200.0,
        "median_abs_cents_error": 1200.0,
        "stability_mad_cents": 8.0,
        "tolerance_cents": 50.0,
        "in_tolerance_ratio": 0.0,
        "target_note": "A4",
        "detected_note": "A3",
    }
    msgs = generate_singing_feedback(summary)
    assert any("oktaf" in m.lower() for m in msgs)
