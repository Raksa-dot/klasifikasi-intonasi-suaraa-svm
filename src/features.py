"""Ekstraksi fitur statistik pitch, vektor fitur SVM 14 dimensi, dan metrik latihan bernyanyi."""

from typing import Any, Dict, Optional
import numpy as np

from src.config import (
    DEFAULT_CENTS_TOLERANCE,
    INTONATION_FEATURE_NAMES,
    MIN_VOICED_FRAMES,
    MIN_VOICED_RATIO,
)
from src.notes import calculate_cents_error, hz_to_note_name
from src.pitch import PitchContour


def compute_relative_semitones(f0_hz: np.ndarray) -> np.ndarray:
    """Konversi F0 valid ke semitone relatif terhadap median pitch rekaman (frame non-voiced tetap NaN)."""
    valid_mask = np.isfinite(f0_hz) & (f0_hz > 0)
    if not np.any(valid_mask):
        return np.full_like(f0_hz, np.nan, dtype=float)
    median_f0 = float(np.median(f0_hz[valid_mask]))
    out = np.full_like(f0_hz, np.nan, dtype=float)
    out[valid_mask] = 12.0 * np.log2(f0_hz[valid_mask] / median_f0)
    return out


def extract_intonation_feature_dict(contour: PitchContour) -> Optional[Dict[str, float]]:
    """Ekstrak 14 fitur numerik kontur intonasi (dalam semitone relatif pembicara) untuk model SVM.

    Mengembalikan None apabila jumlah frame voiced di bawah ambang kelayakan.
    """
    valid_idx = np.where(contour.voiced_mask)[0]
    n_total = max(1, int(contour.f0_hz.size))
    n_valid = int(valid_idx.size)
    voiced_ratio = float(n_valid / n_total)

    if n_valid < MIN_VOICED_FRAMES or voiced_ratio < MIN_VOICED_RATIO:
        return None

    valid_f0 = contour.f0_hz[valid_idx].astype(float)
    valid_times = contour.timestamps[valid_idx].astype(float)
    median_f0 = float(np.median(valid_f0))
    rel_st = 12.0 * np.log2(valid_f0 / median_f0)

    # Normalisasi koordinat waktu voiced ke [0.0, 1.0]
    t_span = float(valid_times[-1] - valid_times[0])
    if t_span > 1e-6:
        t_norm = (valid_times - valid_times[0]) / t_span
        slope_per_sec = float(np.polyfit(valid_times - valid_times[0], rel_st, 1)[0])
        slope_norm = float(np.polyfit(t_norm, rel_st, 1)[0])
    else:
        t_norm = np.linspace(0.0, 1.0, n_valid)
        slope_per_sec = 0.0
        slope_norm = 0.0

    # Bagi frame voiced menjadi 3 segmen proporsional: awal, tengah, akhir
    third = max(1, n_valid // 3)
    seg_start = rel_st[:third]
    seg_mid = rel_st[third : 2 * third] if (2 * third) > third else rel_st
    seg_end = rel_st[-third:]

    start_mean_st = float(np.mean(seg_start))
    mid_mean_st = float(np.mean(seg_mid))
    end_mean_st = float(np.mean(seg_end))

    # Koefisien kelengkungan orde-2 (positif = lembah / FALL_RISE, negatif = bukit / RISE_FALL)
    if n_valid >= 5:
        poly2_coeffs = np.polyfit(t_norm, rel_st, 2)
        poly2_curvature = float(poly2_coeffs[0])
    else:
        poly2_curvature = 0.0

    peak_pos = float(t_norm[int(np.argmax(rel_st))])
    valley_pos = float(t_norm[int(np.argmin(rel_st))])

    p10, p25, p75, p90 = np.percentile(rel_st, [10, 25, 75, 90])

    feature_dict = {
        "pitch_range_semitones": float(np.max(rel_st) - np.min(rel_st)),
        "std_semitones": float(np.std(rel_st)),
        "iqr_semitones": float(p75 - p25),
        "p90_minus_p10_semitones": float(p90 - p10),
        "overall_slope_st_per_sec": slope_per_sec,
        "overall_slope_st_normalized": slope_norm,
        "start_mean_st": start_mean_st,
        "mid_mean_st": mid_mean_st,
        "end_mean_st": end_mean_st,
        "end_minus_start_st": float(end_mean_st - start_mean_st),
        "mid_minus_start_st": float(mid_mean_st - start_mean_st),
        "end_minus_mid_st": float(end_mean_st - mid_mean_st),
        "poly2_curvature": poly2_curvature,
        "peak_minus_valley_pos": float(peak_pos - valley_pos),
    }

    # Pastikan urutan sesuai INTONATION_FEATURE_NAMES
    return {name: float(feature_dict[name]) for name in INTONATION_FEATURE_NAMES}


def summarize_speaking_pitch(contour: PitchContour, duration_sec: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Ringkas fitur F0 untuk latihan berbicara dari frame voiced yang valid."""
    feature_dict = extract_intonation_feature_dict(contour)
    if feature_dict is None:
        return None

    valid_f0 = contour.f0_hz[contour.voiced_mask].astype(float)
    valid_times = contour.timestamps[contour.voiced_mask].astype(float)
    n_valid = int(valid_f0.size)
    n_total = max(1, int(contour.f0_hz.size))

    if duration_sec is None:
        duration_sec = float(contour.timestamps[-1]) if contour.timestamps.size > 0 else 0.0

    hop_sec = float(contour.hop_length / contour.sample_rate)
    voiced_duration_sec = float(n_valid * hop_sec)

    third = max(1, n_valid // 3)
    start_mean_hz = float(np.mean(valid_f0[:third]))
    end_mean_hz = float(np.mean(valid_f0[-third:]))
    start_end_diff_hz = float(end_mean_hz - start_mean_hz)

    summary: Dict[str, Any] = {
        "mean_f0_hz": float(np.mean(valid_f0)),
        "median_f0_hz": float(np.median(valid_f0)),
        "std_f0_hz": float(np.std(valid_f0)),
        "min_f0_hz": float(np.min(valid_f0)),
        "max_f0_hz": float(np.max(valid_f0)),
        "p10_f0_hz": float(np.percentile(valid_f0, 10)),
        "p90_f0_hz": float(np.percentile(valid_f0, 90)),
        "lowest_note": hz_to_note_name(float(np.min(valid_f0))),
        "highest_note": hz_to_note_name(float(np.max(valid_f0))),
        "median_note": hz_to_note_name(float(np.median(valid_f0))),
        "pitch_range_hz": float(np.max(valid_f0) - np.min(valid_f0)),
        "pitch_range_semitones": feature_dict["pitch_range_semitones"],
        "start_end_diff_semitones": feature_dict["end_minus_start_st"],
        "start_end_diff_hz": start_end_diff_hz,
        "slope_semitones_per_sec": feature_dict["overall_slope_st_per_sec"],
        "duration_sec": float(duration_sec),
        "voiced_duration_sec": voiced_duration_sec,
        "voiced_frames": n_valid,
        "total_frames": n_total,
        "voiced_ratio": float(n_valid / n_total),
        "first_voiced_sec": float(valid_times[0]),
        "last_voiced_sec": float(valid_times[-1]),
        "svm_features": feature_dict,
    }
    summary.update(feature_dict)
    return summary


def summarize_singing_pitch(
    contour: PitchContour,
    target_hz: float,
    tolerance_cents: float = DEFAULT_CENTS_TOLERANCE,
    duration_sec: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Hitung ketepatan terhadap nada target dan kestabilan nada secara terpisah."""
    valid_idx = np.where(contour.voiced_mask)[0]
    n_total = max(1, int(contour.f0_hz.size))
    n_valid = int(valid_idx.size)
    voiced_ratio = float(n_valid / n_total)

    if n_valid < MIN_VOICED_FRAMES or voiced_ratio < MIN_VOICED_RATIO or target_hz <= 0:
        return None

    valid_f0 = contour.f0_hz[valid_idx].astype(float)
    cents_errors = np.array([calculate_cents_error(float(f), target_hz) for f in valid_f0], dtype=float)
    median_f0 = float(np.median(valid_f0))

    # Untuk kestabilan nada yang ditahan: gunakan bagian inti stabil (10%..90% frame voiced bila cukup panjang)
    if n_valid >= 20:
        trim_k = max(1, int(0.10 * n_valid))
        core_f0 = valid_f0[trim_k:-trim_k]
    else:
        core_f0 = valid_f0

    core_median_f0 = float(np.median(core_f0))
    cents_from_self_median = 1200.0 * np.log2(core_f0 / core_median_f0)
    mad_stability_cents = float(np.median(np.abs(cents_from_self_median - np.median(cents_from_self_median))))
    std_stability_cents = float(np.std(cents_from_self_median))

    in_tol_mask = np.abs(cents_errors) <= tolerance_cents
    in_tol_ratio = float(np.mean(in_tol_mask))

    hop_sec = float(contour.hop_length / contour.sample_rate)
    voiced_duration_sec = float(n_valid * hop_sec)
    if duration_sec is None:
        duration_sec = float(contour.timestamps[-1]) if contour.timestamps.size > 0 else voiced_duration_sec

    return {
        "target_hz": float(target_hz),
        "target_note": hz_to_note_name(target_hz),
        "median_f0_hz": median_f0,
        "mean_f0_hz": float(np.mean(valid_f0)),
        "min_f0_hz": float(np.min(valid_f0)),
        "max_f0_hz": float(np.max(valid_f0)),
        "detected_note": hz_to_note_name(median_f0),
        "lowest_note": hz_to_note_name(float(np.min(valid_f0))),
        "highest_note": hz_to_note_name(float(np.max(valid_f0))),
        "median_signed_cents_error": float(np.median(cents_errors)),
        "mean_signed_cents_error": float(np.mean(cents_errors)),
        "median_abs_cents_error": float(np.median(np.abs(cents_errors))),
        "mean_abs_cents_error": float(np.mean(np.abs(cents_errors))),
        "stability_mad_cents": mad_stability_cents,
        "stability_std_cents": std_stability_cents,
        "in_tolerance_ratio": in_tol_ratio,
        "in_tolerance_frames": int(np.sum(in_tol_mask)),
        "tolerance_cents": float(tolerance_cents),
        "duration_sec": float(duration_sec),
        "voiced_duration_sec": voiced_duration_sec,
        "voiced_frames": n_valid,
        "total_frames": n_total,
        "voiced_ratio": voiced_ratio,
    }
