"""Unit test untuk validasi perbandingan dua percobaan."""

from src.comparison import can_compare_attempts, summarize_attempt_comparison


def test_reject_mismatched_modes_or_targets() -> None:
    a = {"mode": "SINGING", "exercise_id": "sng_a3", "target": "A3"}
    b = {"mode": "SINGING", "exercise_id": "sng_c4", "target": "C4"}
    ok, _ = can_compare_attempts(a, b)
    assert not ok

    c = {"mode": "SPEAKING", "exercise_id": "spk_1", "target": "RISING"}
    ok_mode, _ = can_compare_attempts(a, c)
    assert not ok_mode


def test_accept_matching_attempts() -> None:
    a = {
        "mode": "SINGING",
        "exercise_id": "sng_a3",
        "target": "A3",
        "summary": {
            "median_abs_cents_error": 35.0,
            "median_signed_cents_error": -35.0,
            "stability_mad_cents": 18.0,
        },
    }
    b = {
        "mode": "SINGING",
        "exercise_id": "sng_a3",
        "target": "A3",
        "summary": {
            "median_abs_cents_error": 12.0,
            "median_signed_cents_error": -12.0,
            "stability_mad_cents": 9.5,
        },
    }
    ok, _ = can_compare_attempts(a, b)
    assert ok
    notes = summarize_attempt_comparison(a, b)
    assert len(notes) >= 2
