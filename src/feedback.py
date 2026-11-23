"""Generator umpan balik (saran latihan) yang dapat ditelusuri ke hasil pengukuran."""

from typing import Any, Dict, List, Optional

from src.config import LABEL_DESCRIPTIONS_ID


def generate_speaking_feedback(
    summary: Optional[Dict[str, Any]],
    target_label: str,
    predicted_label: Optional[str] = None,
) -> List[str]:
    """Buat saran latihan berbicara berdasarkan pengukuran kontur aktual dan hasil SVM bila tersedia."""
    if not summary:
        return [
            "Bagian suara yang memiliki pitch valid belum mencukupi untuk dianalisis. "
            "Silakan rekam ulang di tempat yang cukup tenang dengan suara yang jelas."
        ]

    diff_st = float(summary.get("start_end_diff_semitones", 0.0))
    span_st = float(summary.get("pitch_range_semitones", 0.0))
    mid_start_st = float(summary.get("mid_minus_start_st", 0.0))
    end_mid_st = float(summary.get("end_minus_mid_st", 0.0))
    voiced_pct = float(summary.get("voiced_ratio", 0.0)) * 100.0

    messages: List[str] = [
        f"Pengukuran aktual: perubahan pitch akhir terhadap awal frasa sebesar {diff_st:+.2f} semitone "
        f"(rentang total {span_st:.2f} semitone, cakupan voiced {voiced_pct:.1f}%)."
    ]

    if predicted_label:
        desc = LABEL_DESCRIPTIONS_ID.get(predicted_label, predicted_label)
        if predicted_label == target_label:
            messages.append(
                f"Hasil klasifikasi SVM ({predicted_label} — {desc}) sesuai dengan pola target latihan yang Anda pilih."
            )
        else:
            messages.append(
                f"Hasil klasifikasi SVM mendeteksi pola {predicted_label} ({desc}), "
                f"sedangkan target latihan Anda adalah {target_label}."
            )

    # Saran spesifik yang ditelusuri ke pengukuran segmen
    if target_label == "RISING":
        if diff_st < 1.2:
            messages.append(
                "Saran latihan: Untuk target RISING (Naik), angkat pitch lebih jelas pada suku kata terakhir frasa "
                "(saat ini perubahan akhir–awal masih di bawah +1,2 semitone)."
            )
        else:
            messages.append(
                "Saran latihan: Kenaikan pitch menuju akhir frasa sudah terlihat jelas. Coba ulangi untuk menjaga konsistensi."
            )
    elif target_label == "FALLING":
        if diff_st > -1.2:
            messages.append(
                "Saran latihan: Untuk target FALLING (Turun), turunkan pitch secara bertahap dan jelas pada penutup frasa."
            )
        else:
            messages.append(
                "Saran latihan: Penurunan pitch menuju akhir frasa sudah terukur dengan jelas."
            )
    elif target_label == "LEVEL":
        if span_st > 4.0 or abs(diff_st) > 1.5:
            messages.append(
                "Saran latihan: Untuk target LEVEL (Datar), jaga agar tinggi suara tidak terlalu naik atau turun di akhir frasa."
            )
        else:
            messages.append(
                "Saran latihan: Variasi pitch sepanjang frasa sudah terjaga relatif datar dan stabil."
            )
    elif target_label == "FALL_RISE":
        if not (mid_start_st < -0.5 and end_mid_st > 0.8):
            messages.append(
                f"Saran latihan: Untuk target FALL_RISE, turunkan pitch di tengah frasa (saat ini {mid_start_st:+.2f} ST) "
                f"lalu naikkan kembali di akhir frasa (saat ini {end_mid_st:+.2f} ST)."
            )
        else:
            messages.append("Saran latihan: Pola turun di bagian tengah lalu naik di bagian akhir sudah terbentuk.")
    elif target_label == "RISE_FALL":
        if not (mid_start_st > 0.5 and end_mid_st < -0.8):
            messages.append(
                f"Saran latihan: Untuk target RISE_FALL, beri penekanan pitch naik di tengah frasa (saat ini {mid_start_st:+.2f} ST) "
                f"lalu turunkan pada penutup frasa (saat ini {end_mid_st:+.2f} ST)."
            )
        else:
            messages.append("Saran latihan: Pola naik ke puncak di tengah lalu turun di akhir frasa sudah terbentuk.")

    return messages


def generate_singing_feedback(summary: Optional[Dict[str, Any]]) -> List[str]:
    """Buat saran latihan bernyanyi berdasarkan selisih cents dan kestabilan nada secara terpisah."""
    if not summary:
        return [
            "Bagian suara bernada belum terdeteksi secara memadai. "
            "Silakan merekam ulang dengan menahan vokal (mis. 'aaa') selama 2–4 detik."
        ]

    signed_cents = float(summary.get("median_signed_cents_error", 0.0))
    abs_cents = float(summary.get("median_abs_cents_error", 0.0))
    mad_cents = float(summary.get("stability_mad_cents", 0.0))
    tol_cents = float(summary.get("tolerance_cents", 50.0))
    in_tol_pct = float(summary.get("in_tolerance_ratio", 0.0)) * 100.0
    target_note = str(summary.get("target_note", ""))
    detected_note = str(summary.get("detected_note", ""))

    messages: List[str] = []

    # 1. Evaluasi ketepatan nada (termasuk cek kesalahan oktaf >= 600 cents)
    if abs(signed_cents) >= 600.0:
        direction = "lebih tinggi" if signed_cents > 0 else "lebih rendah"
        messages.append(
            f"Ketepatan nada: Nada yang terdeteksi ({detected_note}) terpaut jauh ({signed_cents:+.1f} cents / "
            f"{direction} hampir satu oktaf) dari target {target_note}. Dengarkan kembali nada acuan sebelum merekam."
        )
    elif abs_cents <= tol_cents:
        messages.append(
            f"Ketepatan nada: Median pitch Anda ({detected_note}, selisih {signed_cents:+.1f} cents) berada di dalam "
            f"parameter toleransi latihan (±{tol_cents:.0f} cents) dengan {in_tol_pct:.1f}% frame valid masuk toleransi."
        )
    elif signed_cents > tol_cents:
        messages.append(
            f"Ketepatan nada: Pitch Anda cenderung lebih tinggi (sharp) sebesar {signed_cents:+.1f} cents terhadap target {target_note}. "
            "Coba bidik nada awal sedikit lebih rendah."
        )
    else:
        messages.append(
            f"Ketepatan nada: Pitch Anda cenderung lebih rendah (flat) sebesar {signed_cents:+.1f} cents terhadap target {target_note}. "
            "Dengarkan nada acuan dan angkat sedikit bidikan pitch Anda."
        )

    # 2. Evaluasi kestabilan nada secara terpisah dari ketepatan
    if mad_cents <= 20.0:
        if abs_cents > tol_cents:
            messages.append(
                f"Kestabilan nada: Nada ditahan dengan stabil (MAD {mad_cents:.1f} cents), "
                "namun posisinya masih meleset dari frekuensi target. Fokuskan percobaan berikutnya pada ketepatan bidikan awal."
            )
        else:
            messages.append(
                f"Kestabilan nada: Pitch ditahan dengan sangat stabil (variasi MAD {mad_cents:.1f} cents)."
            )
    elif mad_cents <= 45.0:
        messages.append(
            f"Kestabilan nada: Variasi pitch selama nada ditahan berada pada tingkat menengah (MAD {mad_cents:.1f} cents). "
            "Jaga aliran napas tetap rata hingga akhir rekaman."
        )
    else:
        messages.append(
            f"Kestabilan nada: Pitch masih cukup berfluktuasi selama nada ditahan (MAD {mad_cents:.1f} cents). "
            "Usahakan menahan satu vokal dengan dorongan napas yang konstan."
        )

    return messages
