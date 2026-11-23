"""Perbandingan dua percobaan dalam sesi latihan yang sama."""

from typing import Any, Dict, List, Tuple


def can_compare_attempts(attempt_a: Dict[str, Any], attempt_b: Dict[str, Any]) -> Tuple[bool, str]:
    """Periksa apakah dua percobaan berasal dari mode, latihan, dan target yang sama."""
    if not attempt_a or not attempt_b:
        return False, "Diperlukan minimal dua percobaan yang valid untuk melakukan perbandingan."
    if attempt_a.get("mode") != attempt_b.get("mode"):
        return False, "Perbandingan tidak setara karena mode latihan berbeda (Berbicara vs Bernyanyi)."
    if attempt_a.get("exercise_id") != attempt_b.get("exercise_id") or attempt_a.get("target") != attempt_b.get("target"):
        return (
            False,
            f"Perbandingan tidak setara: Percobaan A melatih target '{attempt_a.get('target')}' "
            f"sedangkan Percobaan B melatih target '{attempt_b.get('target')}'. "
            "Pilih dua percobaan dari latihan dan target yang sama.",
        )
    return True, "Kedua percobaan memiliki mode, latihan, dan target yang sama."


def summarize_attempt_comparison(attempt_a: Dict[str, Any], attempt_b: Dict[str, Any]) -> List[str]:
    """Buat ringkasan objektif perubahan metrik dari Percobaan A ke Percobaan B."""
    ok, reason = can_compare_attempts(attempt_a, attempt_b)
    if not ok:
        return [reason]

    mode = attempt_a.get("mode", "SPEAKING")
    sum_a = attempt_a.get("summary", {})
    sum_b = attempt_b.get("summary", {})
    notes: List[str] = []

    if mode == "SPEAKING":
        diff_a = float(sum_a.get("start_end_diff_semitones", 0.0))
        diff_b = float(sum_b.get("start_end_diff_semitones", 0.0))
        span_a = float(sum_a.get("pitch_range_semitones", 0.0))
        span_b = float(sum_b.get("pitch_range_semitones", 0.0))
        notes.append(
            f"Perubahan pitch akhir–awal frasa: Percobaan A = {diff_a:+.2f} ST → Percobaan B = {diff_b:+.2f} ST."
        )
        notes.append(
            f"Rentang pitch terdeteksi: Percobaan A = {span_a:.2f} ST → Percobaan B = {span_b:.2f} ST."
        )
        pred_a = attempt_a.get("predicted_label")
        pred_b = attempt_b.get("predicted_label")
        if pred_a or pred_b:
            notes.append(
                f"Prediksi klasifikasi SVM: Percobaan A = {pred_a or 'Belum tersedia'} → "
                f"Percobaan B = {pred_b or 'Belum tersedia'} (Target: {attempt_a.get('target')})."
            )
    else:
        abs_a = float(sum_a.get("median_abs_cents_error", 0.0))
        abs_b = float(sum_b.get("median_abs_cents_error", 0.0))
        signed_a = float(sum_a.get("median_signed_cents_error", 0.0))
        signed_b = float(sum_b.get("median_signed_cents_error", 0.0))
        mad_a = float(sum_a.get("stability_mad_cents", 0.0))
        mad_b = float(sum_b.get("stability_mad_cents", 0.0))
        notes.append(
            f"Kedekatan terhadap nada target ({attempt_a.get('target')}): "
            f"selisih bertanda {signed_a:+.1f} cents (|error| {abs_a:.1f} cents) pada Percobaan A → "
            f"{signed_b:+.1f} cents (|error| {abs_b:.1f} cents) pada Percobaan B."
        )
        notes.append(
            f"Variasi kestabilan nada (MAD): Percobaan A = {mad_a:.1f} cents → Percobaan B = {mad_b:.1f} cents."
        )

    return notes
