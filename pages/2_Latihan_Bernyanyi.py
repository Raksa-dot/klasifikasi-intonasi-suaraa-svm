"""Halaman 2: Latihan Bernyanyi (Ikuti dan Tahan Satu Nada Acuan)."""

from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

from src.audio_io import encode_wav_bytes, read_audio_bytes
from src.config import A4_REFERENCE_HZ, DEFAULT_CENTS_TOLERANCE
from src.exercises import load_exercises
from src.features import compute_relative_semitones, summarize_singing_pitch
from src.feedback import generate_singing_feedback
from src.notes import calculate_cents_error
from src.pitch import extract_pitch_contour
from src.preprocessing import preprocess_audio
from src.reference_tones import synthesize_reference_tone_wav
from src.ui_components import (
    clear_current_input_and_result,
    init_session_state,
    render_live_pitch_monitor,
    render_shared_audio_input,
    save_attempt_to_session,
)
from src.validation import validate_raw_signal
from src.visualization import create_pitch_contour_figure, create_waveform_figure

st.set_page_config(page_title="Latihan Bernyanyi — Voice Analyzer", page_icon="🎵", layout="wide")
init_session_state()

st.title("🎵 Latihan Bernyanyi — Ikuti dan Tahan Satu Nada Acuan")
st.write(
    f"Pilih nada target yang nyaman bagi suara Anda (standar tuning **A4 = {A4_REFERENCE_HZ:.1f} Hz**), "
    "dengarkan nada acuan, lalu rekam suara Anda saat mengikuti dan menahan nada tersebut."
)

singing_exercises = load_exercises().get("singing_exercises", [])

col_cfg1, col_cfg2, col_cfg3 = st.columns([2, 1, 1])
with col_cfg1:
    labels = [f"{ex['note_name']} — {ex['target_hz']:.2f} Hz" for ex in singing_exercises]
    selected_idx = st.selectbox("Pilih Nada Target Latihan:", range(len(singing_exercises)), format_func=lambda i: labels[i])
    selected_ex = singing_exercises[selected_idx]

with col_cfg2:
    practice_type = st.selectbox(
        "Jenis Latihan MVP:",
        ["Ikuti Satu Nada", "Tahan Nada (Sustained Note)"],
    )

with col_cfg3:
    tolerance_cents = float(
        st.selectbox(
            "Parameter Toleransi Latihan:",
            [50, 35, 25],
            index=0,
            format_func=lambda v: f"±{v} cents (Parameter Evaluasi)",
        )
    )

target_note = selected_ex["note_name"]
target_hz = float(selected_ex["target_hz"])
exercise_id = f"{selected_ex['id']}_{practice_type.split()[0].lower()}"

# Panel Nada Acuan Lokal
st.markdown("### 🔊 1. Nada Acuan & Instruksi Latihan")
col_ref1, col_ref2 = st.columns([1, 1])
with col_ref1:
    st.info(
        f"**Nada Target:** `{target_note}` (**{target_hz:.2f} Hz**, tuning A4 = {A4_REFERENCE_HZ:.0f} Hz)\n\n"
        f"**Mode Latihan:** {practice_type}\n\n"
        f"**Instruksi:** {selected_ex['instruction']} Tahan vokal (misalnya *'aaa'* atau *'uuu'*) selama sekitar "
        f"**{selected_ex.get('target_duration_sec', 3.0):.0f} detik**."
    )
with col_ref2:
    st.markdown("**Pemutar Nada Acuan (Disintesis Secara Lokal):**")
    tone_bytes = synthesize_reference_tone_wav(target_hz, duration_sec=3.0)
    st.audio(tone_bytes, format="audio/wav")
    st.caption(
        "⚠️ **Penting:** Hentikan pemutaran nada acuan sebelum mulai merekam, atau gunakan **headphone/earphone** "
        "agar bunyi nada acuan dari speaker tidak ikut tertangkap oleh mikrofon."
    )

# Monitor Tuner Live Real-Time
with st.expander(f"⚡ Buka Live Pitch Tuner Real-Time (Bidik Nada Target {target_note} — {target_hz:.2f} Hz)", expanded=False):
    render_live_pitch_monitor(mode="SINGING", target_hz=target_hz, target_label=target_note)

st.divider()

# Input Audio Bersama
audio_bytes, filename_hint, analyze_clicked = render_shared_audio_input(
    mode="SINGING",
    exercise_id=exercise_id,
    target_key=f"{target_note}:{target_hz:.2f}:{tolerance_cents:.0f}",
)

if analyze_clicked and audio_bytes is not None:
    with st.spinner("Memvalidasi sinyal, mengekstraksi kontur F0, dan menghitung selisih cents serta kestabilan nada..."):
        try:
            raw_samples, orig_sr = read_audio_bytes(audio_bytes, filename_hint=filename_hint)
            val_res = validate_raw_signal(raw_samples, orig_sr, file_size_bytes=len(audio_bytes))
            pre = preprocess_audio(raw_samples, orig_sr)
            contour = extract_pitch_contour(
                pre.samples,
                sample_rate=pre.sample_rate,
                mode="SINGING",
                time_offset_sec=pre.time_offset_sec,
            )
            summary = summarize_singing_pitch(
                contour,
                target_hz=target_hz,
                tolerance_cents=tolerance_cents,
                duration_sec=val_res.duration_sec,
            )
            feedback_msgs = generate_singing_feedback(summary)
            wav_play_bytes = encode_wav_bytes(pre.samples, pre.sample_rate)

            # Hitung kontur selisih cents per frame untuk grafik
            cents_contour = np.full_like(contour.f0_hz, np.nan, dtype=float)
            for idx_f, f_val in enumerate(contour.f0_hz):
                if np.isfinite(f_val) and f_val > 0:
                    c_err = calculate_cents_error(float(f_val), target_hz)
                    if c_err is not None:
                        cents_contour[idx_f] = c_err

            analysis_payload = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "mode": "SINGING",
                "exercise_id": exercise_id,
                "target": target_note,
                "target_hz": target_hz,
                "practice_type": practice_type,
                "validation": val_res,
                "preprocessed": pre,
                "contour": contour,
                "cents_contour": cents_contour,
                "relative_semitones": compute_relative_semitones(contour.f0_hz),
                "summary": summary,
                "predicted_label": None,
                "feedback": feedback_msgs,
                "audio_wav_bytes": wav_play_bytes,
            }
            st.session_state["current_analysis"] = analysis_payload

            if val_res.is_valid and summary is not None:
                att_num = save_attempt_to_session(analysis_payload)
                st.toast(f"Hasil latihan bernyanyi disimpan sebagai Percobaan #{att_num} di sesi ini.")
        except Exception as exc:
            st.error(f"Gagal memproses file audio: {exc}")

current = st.session_state.get("current_analysis")
if current and current.get("mode") == "SINGING":
    st.divider()
    st.markdown("### 📈 2. Hasil Analisis Latihan Bernyanyi (Ketepatan & Kestabilan Terpisah)")

    val_res = current["validation"]
    summary = current["summary"]
    contour = current["contour"]

    if val_res.warnings:
        for w in val_res.warnings:
            st.warning(w)

    if not val_res.is_valid or summary is None:
        st.error("Rekaman belum memenuhi syarat untuk penilaian nada yang valid:")
        for reason in val_res.reasons:
            st.markdown(f"- {reason}")
        if summary is None:
            st.markdown(
                "- Bagian suara yang memiliki pitch valid (*voiced frames*) terlalu sedikit. "
                "Silakan merekam ulang dengan menahan nada lebih jelas."
            )
    else:
        signed_c = float(summary["median_signed_cents_error"])
        abs_c = float(summary["median_abs_cents_error"])
        mad_c = float(summary["stability_mad_cents"])
        in_tol_pct = float(summary["in_tolerance_ratio"]) * 100.0
        direction_txt = "Lebih Tinggi (+)" if signed_c > 0.5 else ("Lebih Rendah (-)" if signed_c < -0.5 else "Tepat pada Target")

        # Tampilkan Ketepatan Nada dan Kestabilan Nada secara TERPISAH
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Nada Target vs Terdeteksi",
            f"{summary['target_note']} → {summary['detected_note']}",
            f"Median: {summary['median_f0_hz']:.1f} Hz (Target {summary['target_hz']:.1f} Hz)",
            delta_color="off",
        )
        c2.metric(
            "Ketepatan Nada (Selisih Bertanda)",
            f"{signed_c:+.1f} cents",
            f"{direction_txt} | Median |Error|: {abs_c:.1f} cents",
            delta_color="off",
        )
        c3.metric(
            "Kestabilan Nada (Variasi MAD)",
            f"{mad_c:.1f} cents",
            f"Std Segmen Inti: {summary['stability_std_cents']:.1f} cents",
            delta_color="off",
            help="Median Absolute Deviation (cents) terhadap median pitch segmen itu sendiri (terpisah dari selisih terhadap target).",
        )
        c4.metric(
            f"Dalam Toleransi (±{summary['tolerance_cents']:.0f} c)",
            f"{in_tol_pct:.1f}% frame valid",
            f"Voiced: {summary['voiced_duration_sec']:.2f} dtk ({summary['voiced_ratio']*100:.1f}%)",
            delta_color="off",
        )

        # Grafik Kontur Pitch (Hz) dengan Garis Nada Target & Zona Toleransi
        fig_hz = create_pitch_contour_figure(
            timestamps=contour.timestamps,
            f0_values=contour.f0_hz,
            y_label="Pitch / F0 (Hz)",
            title=f"Kontur Pitch (Hz) vs Garis Nada Target {summary['target_note']} ({summary['target_hz']:.2f} Hz)",
            target_hz=float(summary["target_hz"]),
            target_label=f"Target {summary['target_note']} ({summary['target_hz']:.2f} Hz)",
            tolerance_cents=float(summary["tolerance_cents"]),
        )
        st.plotly_chart(fig_hz)

        col_fb1, col_fb2 = st.columns([3, 2])
        with col_fb1:
            st.markdown("#### 💡 Saran Latihan Bernyanyi (Berdasarkan Pengukuran)")
            for msg in current["feedback"]:
                st.markdown(f"- {msg}")
            st.caption(
                "Catatan: Ketepatan nada (kedekatan terhadap target) dan kestabilan nada (variasi selama nada ditahan) "
                "adalah dua besaran yang terpisah. Nada dapat sangat stabil sekaligus meleset dari target."
            )
        with col_fb2:
            st.markdown("#### 🔄 Langkah Berikutnya")
            if st.button("🎙️ Coba Lagi Nada Ini"):
                clear_current_input_and_result()
                st.rerun()
            st.page_link(
                "pages/3_Bandingkan_Percobaan.py",
                label="📊 Bandingkan Percobaan di Sesi Ini →",
            )

        with st.expander("🔬 Buka Detail Teknis (Grafik Selisih Cents per Waktu, Rentang Terdeteksi & Waveform)"):
            st.write(
                f"- **Rentang Nada Terdeteksi dalam Rekaman Ini:** `{summary['lowest_note']}` (`{summary['min_f0_hz']:.2f} Hz`) "
                f"hingga `{summary['highest_note']}` (`{summary['max_f0_hz']:.2f} Hz`). "
                "*(Ini adalah rentang pada rekaman yang dianalisis, bukan keseluruhan kemampuan vokal Anda)*."
            )
            st.write(
                f"- **Denominator Frame Valid:** `{summary['in_tolerance_frames']}` dari `{summary['voiced_frames']}` frame voiced "
                f"berada dalam toleransi `±{summary['tolerance_cents']:.0f} cents` (total frame rekaman: `{summary['total_frames']}`)."
            )

            fig_cents = create_pitch_contour_figure(
                timestamps=contour.timestamps,
                f0_values=current["cents_contour"],
                y_label="Selisih terhadap Target (cents)",
                title=f"Selisih Bertanda dalam Cents terhadap Target {summary['target_note']} (0 cents = Tepat pada Target)",
            )
            fig_cents.add_hline(y=0.0, line_dash="dash", line_color="#dc2626", annotation_text="Target (0 cents)")
            st.plotly_chart(fig_cents)

            fig_wave = create_waveform_figure(
                current["preprocessed"].samples,
                current["preprocessed"].sample_rate,
                time_offset_sec=current["preprocessed"].time_offset_sec,
            )
            st.plotly_chart(fig_wave)
