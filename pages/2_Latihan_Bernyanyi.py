"""Halaman 2: Latihan Bernyanyi (Ikuti dan Tahan Satu Nada Acuan)."""

from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

from src.audio_io import encode_wav_bytes, read_audio_bytes
from src.comparison import summarize_attempt_comparison
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
    inject_custom_css,
    render_live_pitch_monitor,
    render_shared_audio_input,
    save_attempt_to_session,
)
from src.validation import validate_raw_signal
from src.visualization import (
    create_comparison_overlay_figure,
    create_pitch_contour_figure,
    create_waveform_figure,
)

st.set_page_config(page_title="Latihan Bernyanyi — Voice Analyzer", page_icon="🎙️", layout="wide")
inject_custom_css()
init_session_state()

st.markdown(
    f"""
    <div style="margin-bottom: 20px;">
      <span class="va-header-tag">Modul 02 • Ketepatan & Kestabilan Nada</span>
      <h1 style="margin: 0;">Latihan Nada Bernyanyi</h1>
      <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px;">
        Dengarkan acuan nada lokal (standar A4 = {A4_REFERENCE_HZ:.0f} Hz), pertahankan vokal stabil, dan evaluasi ketepatan cents serta variasi MAD secara terpisah.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

singing_exercises = load_exercises().get("singing_exercises", [])

col_cfg1, col_cfg2, col_cfg3 = st.columns([2, 1, 1])
with col_cfg1:
    labels = [f"{ex['note_name']} — {ex['target_hz']:.2f} Hz ({ex.get('title', '')})" for ex in singing_exercises]
    selected_idx = st.selectbox("Pilih Nada Target:", range(len(singing_exercises)), format_func=lambda i: labels[i])
    selected_ex = singing_exercises[selected_idx]

with col_cfg2:
    practice_type = st.selectbox(
        "Kategori Latihan:",
        ["Ikuti Satu Nada", "Tahan Nada Stabil (Sustained)"],
    )

with col_cfg3:
    tolerance_cents = float(
        st.selectbox(
            "Parameter Toleransi:",
            [50, 35, 25],
            index=0,
            format_func=lambda v: f"±{v} cents",
        )
    )

target_note = selected_ex["note_name"]
target_hz = float(selected_ex["target_hz"])
exercise_id = f"{selected_ex['id']}_{practice_type.split()[0].lower()}"

# Panel Nada Acuan & Panduan
st.markdown("### 01. Nada Acuan & Konfigurasi Target")
col_ref1, col_ref2 = st.columns([1, 1])
with col_ref1:
    st.markdown(
        f"""
        <div class="va-card">
          <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #34d399; letter-spacing: 0.05em; margin-bottom: 4px;">Target Nada Vokal</div>
          <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">{target_note} ({target_hz:.2f} Hz)</div>
          <div style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 8px;">Mode: {practice_type} • Batas Toleransi: &plusmn;{tolerance_cents:.0f} cents</div>
          <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.45;"><strong>Petunjuk:</strong> {selected_ex['instruction']} Tahan nada vokal selama ~{selected_ex.get('target_duration_sec', 3.0):.0f} detik.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_ref2:
    st.markdown("**Generator Nada Acuan Lokal (Pure Sine Wave):**")
    tone_bytes = synthesize_reference_tone_wav(target_hz, duration_sec=3.0)
    st.audio(tone_bytes, format="audio/wav")
    st.caption(
        "Catatan Akustik: Hentikan nada acuan sebelum merekam atau gunakan earphone agar frekuensi acuan tidak tertangkap mikrofon."
    )

# Monitor Tuner Real-Time
with st.expander(f"Buka Tuner Frekuensi Waktu-Nyata (Target: {target_note} — {target_hz:.2f} Hz)", expanded=False):
    render_live_pitch_monitor(mode="SINGING", target_hz=target_hz, target_label=target_note)

st.divider()

# Input Audio Bersama
audio_bytes, filename_hint, analyze_clicked = render_shared_audio_input(
    mode="SINGING",
    exercise_id=exercise_id,
    target_key=f"{target_note}:{target_hz:.2f}:{tolerance_cents:.0f}",
)

if analyze_clicked and audio_bytes is not None:
    with st.spinner("Memvalidasi sinyal audio, mengekstraksi pitch F0, dan menghitung metrik kestabilan..."):
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
                st.toast(f"Hasil latihan tersimpan sebagai Rekaman #{att_num}.")
        except Exception as exc:
            st.error(f"Gagal memproses sinyal audio: {exc}")

current = st.session_state.get("current_analysis")
if current and current.get("mode") == "SINGING":
    st.divider()
    st.markdown("### 02. Hasil Evaluasi Ketepatan & Kestabilan Vokal")

    val_res = current["validation"]
    summary = current["summary"]
    contour = current["contour"]

    if val_res.warnings:
        for w in val_res.warnings:
            st.warning(w)

    if not val_res.is_valid or summary is None:
        st.error("Rekaman belum memenuhi syarat analisis pitch bernyanyi:")
        for reason in val_res.reasons:
            st.markdown(f"- {reason}")
        if summary is None:
            st.markdown(
                "- Proporsi frame voiced tidak memadai. Silakan rekam ulang dengan menahan nada vokal secara lebih jelas."
            )
    else:
        signed_c = float(summary["median_signed_cents_error"])
        abs_c = float(summary["median_abs_cents_error"])
        mad_c = float(summary["stability_mad_cents"])
        in_tol_pct = float(summary["in_tolerance_ratio"]) * 100.0
        direction_txt = "Lebih Tinggi (+)" if signed_c > 0.5 else ("Lebih Rendah (-)" if signed_c < -0.5 else "Presisi pada Target")

        # 4 Metrik Utama: Ketepatan vs Kestabilan Terpisah
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Target vs Terdeteksi",
            f"{summary['target_note']} → {summary['detected_note']}",
            f"{summary['median_f0_hz']:.1f} Hz (Target {summary['target_hz']:.1f} Hz)",
            delta_color="off",
        )
        c2.metric(
            "Ketepatan Nada (Cents)",
            f"{signed_c:+.1f} c",
            f"{direction_txt} • Abs: {abs_c:.1f} c",
            delta_color="off",
        )
        c3.metric(
            "Kestabilan Nada (MAD)",
            f"{mad_c:.1f} c",
            f"Std Inti: {summary['stability_std_cents']:.1f} c",
            delta_color="off",
            help="Median Absolute Deviation (cents) terhadap median pitch segmen itu sendiri (terpisah dari selisih ke target).",
        )
        c4.metric(
            f"Akurasi Toleransi (±{summary['tolerance_cents']:.0f} c)",
            f"{in_tol_pct:.1f}%",
            f"Voiced: {summary['voiced_duration_sec']:.2f} detik",
            delta_color="off",
        )

        # Grafik Kontur Pitch (Hz) vs Garis Target
        fig_hz = create_pitch_contour_figure(
            timestamps=contour.timestamps,
            f0_values=contour.f0_hz,
            y_label="Frekuensi Dasar F0 (Hz)",
            title=f"Kontur Pitch F0 vs Nada Target {summary['target_note']} ({summary['target_hz']:.2f} Hz)",
            target_hz=float(summary["target_hz"]),
            target_label=f"Target {summary['target_note']} ({summary['target_hz']:.2f} Hz)",
            tolerance_cents=float(summary["tolerance_cents"]),
        )
        st.plotly_chart(fig_hz)

        col_fb1, col_fb2 = st.columns([3, 2])
        with col_fb1:
            st.markdown("#### Umpan Balik Latihan Bernyanyi")
            for msg in current["feedback"]:
                st.markdown(f"- {msg}")
            st.caption(
                "Catatan Metrik: Ketepatan nada (kedekatan terhadap target) dan kestabilan nada (variasi MAD selama nada ditahan) "
                "merupakan dua besaran terpisah. Suara dapat sangat stabil namun meleset dari target."
            )
        with col_fb2:
            st.markdown("#### Langkah Berikutnya")
            if st.button("Ulangi Nada Ini"):
                clear_current_input_and_result()
                st.rerun()
            st.page_link(
                "pages/3_Bandingkan_Percobaan.py",
                label="Buka Komparasi Percobaan Sesi →",
            )

        with st.expander("Detail Pengukuran Selisih Cents & Karakteristik Vokal"):
            st.write(
                f"- **Rentang Terdeteksi:** `{summary['lowest_note']}` (`{summary['min_f0_hz']:.1f} Hz`) "
                f"hingga `{summary['highest_note']}` (`{summary['max_f0_hz']:.1f} Hz`). "
                "*(Rentang pada rekaman yang diuji, bukan keseluruhan jangkauan vokal biologis)*."
            )
            st.write(
                f"- **Cakupan Frame:** `{summary['in_tolerance_frames']}` dari `{summary['voiced_frames']}` frame voiced "
                f"berada dalam zona toleransi `±{summary['tolerance_cents']:.0f} cents`."
            )

            fig_cents = create_pitch_contour_figure(
                timestamps=contour.timestamps,
                f0_values=current["cents_contour"],
                y_label="Deviasi terhadap Target (Cents)",
                title=f"Deviasi Bertanda dalam Cents terhadap Target {summary['target_note']} (0 cents = Presisi Target)",
            )
            fig_cents.add_hline(y=0.0, line_dash="dash", line_color="#f43f5e", annotation_text="Target (0 cents)")
            st.plotly_chart(fig_cents)

            fig_wave = create_waveform_figure(
                current["preprocessed"].samples,
                current["preprocessed"].sample_rate,
                time_offset_sec=current["preprocessed"].time_offset_sec,
            )
            st.plotly_chart(fig_wave)

        # 03. Komparasi Langsung Percobaan Bernyanyi (Jika ada >= 2 percobaan pada nada target yang sama)
        matching_attempts = [
            a for a in st.session_state.get("attempts", [])
            if a.get("mode") == "SINGING" and a.get("target") == current["target"]
        ]
        if len(matching_attempts) >= 2:
            st.divider()
            st.markdown("### 03. Komparasi Langsung dengan Percobaan Sebelumnya")
            att_prev = matching_attempts[-2]
            att_curr = matching_attempts[-1]

            st.caption(
                f"Membandingkan **{att_prev['attempt_id']}** ({att_prev['timestamp']}) dengan "
                f"**{att_curr['attempt_id']}** ({att_curr['timestamp']}) pada nada target **{current['target']} ({target_hz:.2f} Hz)**."
            )

            fig_cmp = create_comparison_overlay_figure(
                time_a=att_prev["contour"].timestamps,
                val_a=att_prev["contour"].f0_hz,
                label_a=f"{att_prev['attempt_id']} (Sebelumnya)",
                time_b=att_curr["contour"].timestamps,
                val_b=att_curr["contour"].f0_hz,
                label_b=f"{att_curr['attempt_id']} (Terbaru)",
                normalize_time=False,
                y_label="Frekuensi Dasar / F0 (Hz)",
                title=f"Perbandingan Kontur Pitch — {att_prev['attempt_id']} vs {att_curr['attempt_id']} (Target: {current['target']} — {target_hz:.2f} Hz)",
                target_line_y=target_hz,
                target_line_label=f"Target {current['target']} ({target_hz:.2f} Hz)",
            )
            st.plotly_chart(fig_cmp)

            sum_prev = att_prev.get("summary", {})
            sum_curr = att_curr.get("summary", {})
            cmp_df = pd.DataFrame(
                [
                    {
                        "Aspek Penilaian Terpisah": "Ketepatan: Median Selisih Bertanda (cents)",
                        att_prev["attempt_id"]: f"{sum_prev.get('median_signed_cents_error', 0.0):+.1f} cents",
                        att_curr["attempt_id"]: f"{sum_curr.get('median_signed_cents_error', 0.0):+.1f} cents",
                    },
                    {
                        "Aspek Penilaian Terpisah": "Ketepatan: Median Selisih Absolut (|cents|)",
                        att_prev["attempt_id"]: f"{sum_prev.get('median_abs_cents_error', 0.0):.1f} cents",
                        att_curr["attempt_id"]: f"{sum_curr.get('median_abs_cents_error', 0.0):.1f} cents",
                    },
                    {
                        "Aspek Penilaian Terpisah": "Kestabilan: Variasi MAD Segmen Inti (cents)",
                        att_prev["attempt_id"]: f"{sum_prev.get('stability_mad_cents', 0.0):.1f} cents",
                        att_curr["attempt_id"]: f"{sum_curr.get('stability_mad_cents', 0.0):.1f} cents",
                    },
                    {
                        "Aspek Penilaian Terpisah": f"Proporsi Frame dalam Toleransi (±{sum_curr.get('tolerance_cents', 50.0):.0f} c)",
                        att_prev["attempt_id"]: f"{sum_prev.get('in_tolerance_ratio', 0.0)*100:.1f}%",
                        att_curr["attempt_id"]: f"{sum_curr.get('in_tolerance_ratio', 0.0)*100:.1f}%",
                    },
                    {
                        "Aspek Penilaian Terpisah": "Cakupan Data Valid (Voiced)",
                        att_prev["attempt_id"]: f"{sum_prev.get('voiced_duration_sec', 0.0):.2f} dtk ({sum_prev.get('voiced_ratio', 0.0)*100:.1f}%)",
                        att_curr["attempt_id"]: f"{sum_curr.get('voiced_duration_sec', 0.0):.2f} dtk ({sum_curr.get('voiced_ratio', 0.0)*100:.1f}%)",
                    },
                ]
            )
            st.table(cmp_df)

            st.markdown("#### Ringkasan Perubahan:")
            for note in summarize_attempt_comparison(att_prev, att_curr):
                st.markdown(f"- {note}")
        elif len(matching_attempts) == 1:
            st.info(
                f"💡 **Tips Komparasi Langsung:** Rekaman pertama Anda tersimpan sebagai **{matching_attempts[0]['attempt_id']}**. "
                f"Tekan **Ulangi Nada Ini** atau **Rekam Ulang** dan rekam percobaan kedua pada nada {current['target']} untuk melihat grafik overlay perbandingan langsung di halaman ini!"
            )
