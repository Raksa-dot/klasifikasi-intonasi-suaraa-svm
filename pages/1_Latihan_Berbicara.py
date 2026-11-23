"""Halaman 1: Latihan Berbicara (Pola Intonasi Satu Frasa Pendek)."""

from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

from src.audio_io import encode_wav_bytes, read_audio_bytes
from src.comparison import summarize_attempt_comparison
from src.config import INTONATION_LABELS, LABEL_DESCRIPTIONS_ID
from src.exercises import load_exercises
from src.features import compute_relative_semitones, summarize_speaking_pitch
from src.feedback import generate_speaking_feedback
from src.pitch import extract_pitch_contour
from src.prediction import check_intonation_model_status, predict_intonation_pattern
from src.preprocessing import preprocess_audio
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
    create_target_guide_figure,
    create_waveform_figure,
)

st.set_page_config(page_title="Latihan Berbicara — Voice Analyzer", page_icon="🎙️", layout="wide")
inject_custom_css()
init_session_state()

st.markdown(
    """
    <div style="margin-bottom: 20px;">
      <span class="va-header-tag">Modul 01 • Pola Prosodi Ujaran</span>
      <h1 style="margin: 0;">Latihan Intonasi Berbicara</h1>
      <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px;">
        Evaluasi perubahan frekuensi dasar (F0) sepanjang satu frasa pendek dan klasifikasi pola intonasi dengan Support Vector Machine.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

exercises_list = load_exercises().get("speaking_exercises", [])

col_sel1, col_sel2 = st.columns([1, 1])
with col_sel1:
    source_mode = st.radio(
        "Pilihan Materi Latihan:",
        ["Katalog Frasa Latihan", "Kustomisasi Frasa Mandiri"],
        horizontal=True,
    )

if source_mode == "Katalog Frasa Latihan" and exercises_list:
    with col_sel2:
        options = [f"[{ex['target_label']}] \"{ex['phrase_text']}\" — {ex['title']}" for ex in exercises_list]
        idx = st.selectbox("Pilih Frasa & Target Pola:", range(len(exercises_list)), format_func=lambda i: options[i])
    selected_ex = exercises_list[idx]
    exercise_id = selected_ex["id"]
    target_label = selected_ex["target_label"]
    phrase_text = selected_ex["phrase_text"]
    instruction_text = selected_ex["instruction"]
else:
    with col_sel2:
        target_label = st.selectbox(
            "Target Pola Intonasi:",
            list(INTONATION_LABELS),
            format_func=lambda k: f"{k} — {LABEL_DESCRIPTIONS_ID.get(k, k)}",
        )
    custom_phrase = st.text_input(
        "Teks Satu Frasa Pendek (maks. 60 karakter):",
        value="Sudah siap mulai sekarang?",
        max_chars=60,
    ).strip()
    phrase_text = custom_phrase or "Sudah siap mulai?"
    exercise_id = f"custom_spk_{target_label.lower()}"
    instruction_text = f"Ucapkan satu frasa pendek \"{phrase_text}\" dengan intonasi {LABEL_DESCRIPTIONS_ID.get(target_label, target_label)}."

# Panel Target & Panduan Visual
st.markdown("### 01. Konfigurasi Sasaran & Panduan Kontur")
col_t1, col_t2 = st.columns([1, 1])
with col_t1:
    st.markdown(
        f"""
        <div class="va-card">
          <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #38bdf8; letter-spacing: 0.05em; margin-bottom: 4px;">Parameter Latihan Aktif</div>
          <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">Target: {target_label}</div>
          <div style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 10px;"><strong>Frasa:</strong> &ldquo;{phrase_text}&rdquo;</div>
          <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.45;"><strong>Petunjuk:</strong> {instruction_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_t2:
    st.plotly_chart(create_target_guide_figure(target_label))

# Oscilloscope Real-Time Browser
with st.expander("Buka Oscilloscope Pitch Waktu-Nyata (Panduan Visual Saat Bersuara)", expanded=False):
    render_live_pitch_monitor(mode="SPEAKING", target_hz=None, target_label=target_label)

st.divider()

# Modul Input Audio Bersama
audio_bytes, filename_hint, analyze_clicked = render_shared_audio_input(
    mode="SPEAKING",
    exercise_id=exercise_id,
    target_key=f"{target_label}:{phrase_text}",
)

if analyze_clicked and audio_bytes is not None:
    with st.spinner("Memvalidasi kualitas sinyal audio dan mengekstraksi kontur F0..."):
        try:
            raw_samples, orig_sr = read_audio_bytes(audio_bytes, filename_hint=filename_hint)
            val_res = validate_raw_signal(raw_samples, orig_sr, file_size_bytes=len(audio_bytes))
            pre = preprocess_audio(raw_samples, orig_sr)
            contour = extract_pitch_contour(
                pre.samples,
                sample_rate=pre.sample_rate,
                mode="SPEAKING",
                time_offset_sec=pre.time_offset_sec,
            )
            summary = summarize_speaking_pitch(contour, duration_sec=val_res.duration_sec)
            rel_semitones = compute_relative_semitones(contour.f0_hz)
            pred_res = predict_intonation_pattern(summary.get("svm_features") if summary else None)
            feedback_msgs = generate_speaking_feedback(
                summary,
                target_label=target_label,
                predicted_label=pred_res.predicted_label if pred_res.is_available else None,
            )
            wav_play_bytes = encode_wav_bytes(pre.samples, pre.sample_rate)

            analysis_payload = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "mode": "SPEAKING",
                "exercise_id": exercise_id,
                "target": target_label,
                "phrase_text": phrase_text,
                "validation": val_res,
                "preprocessed": pre,
                "contour": contour,
                "relative_semitones": rel_semitones,
                "summary": summary,
                "predicted_label": pred_res.predicted_label if pred_res.is_available else None,
                "prediction_result": pred_res,
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
if current and current.get("mode") == "SPEAKING":
    st.divider()
    st.markdown("### 02. Hasil Analisis Kontur & Klasifikasi Intonasi")

    val_res = current["validation"]
    summary = current["summary"]
    contour = current["contour"]
    pred_res = current.get("prediction_result")

    if val_res.warnings:
        for w in val_res.warnings:
            st.warning(w)

    if not val_res.is_valid or summary is None:
        st.error("Sinyal audio belum memenuhi ambang kelayakan analisis pitch:")
        for reason in val_res.reasons:
            st.markdown(f"- {reason}")
        if summary is None:
            st.markdown(
                "- Proporsi frame bersuara (voiced frames) terlalu sedikit. Silakan rekam ulang dengan pengucapan yang lebih jelas."
            )
    else:
        # 4 Ringkasan KPI Intonasi
        k1, k2, k3, k4 = st.columns(4)
        diff_st = float(summary.get("start_end_diff_semitones", 0.0))
        diff_hz = float(summary.get("start_end_diff_hz", 0.0))
        k1.metric(
            "Perubahan Akhir - Awal",
            f"{diff_st:+.2f} ST",
            f"{diff_hz:+.1f} Hz",
            delta_color="off",
        )
        k2.metric(
            "Rentang Pitch Valid",
            f"{summary.get('pitch_range_semitones', 0.0):.2f} ST",
            f"{summary.get('min_f0_hz', 0.0):.0f} – {summary.get('max_f0_hz', 0.0):.0f} Hz",
            delta_color="off",
        )
        k3.metric(
            "Median Pitch",
            f"{summary.get('median_f0_hz', 0.0):.1f} Hz",
            f"Notasi {summary.get('median_note', '—')}",
            delta_color="off",
        )
        k4.metric(
            "Cakupan Frame Voiced",
            f"{summary.get('voiced_ratio', 0.0)*100:.1f}%",
            f"{summary.get('voiced_duration_sec', 0.0):.2f} detik",
            delta_color="off",
        )

        # Grafik Kontur Pitch (Hz)
        fig_f0 = create_pitch_contour_figure(
            timestamps=contour.timestamps,
            f0_values=contour.f0_hz,
            y_label="Frekuensi Dasar F0 (Hz)",
            title=f"Kontur Pitch F0 terhadap Waktu — \"{current['phrase_text']}\"",
        )
        st.plotly_chart(fig_f0)

        # Hasil Klasifikasi SVM vs Feedback
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown("#### Klasifikasi Pola Intonasi (SVM)")
            if pred_res and pred_res.is_available and pred_res.predicted_label:
                matches = pred_res.predicted_label == current["target"]
                badge_bg = "rgba(16, 185, 129, 0.12)" if matches else "rgba(245, 158, 11, 0.12)"
                badge_border = "rgba(16, 185, 129, 0.35)" if matches else "rgba(245, 158, 11, 0.35)"
                badge_text_color = "#34d399" if matches else "#fbbf24"
                match_text = "Sesuai dengan Pola Target" if matches else f"Berbeda dari Target ({current['target']})"

                st.markdown(
                    f"""
                    <div style="border: 1px solid {badge_border}; background: {badge_bg}; padding: 16px 20px; border-radius: 8px;">
                      <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; font-weight: 600;">Prediksi Model {pred_res.model_version}</div>
                      <div style="font-size: 24px; font-weight: 700; color: #f8fafc; margin: 4px 0;">{pred_res.predicted_label}</div>
                      <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 6px;">{LABEL_DESCRIPTIONS_ID.get(pred_res.predicted_label, '')}</div>
                      <div style="font-size: 12px; font-weight: 600; color: {badge_text_color};">Status: {match_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info(pred_res.status_message if pred_res else "Klasifikasi intonasi belum tersedia.")

        with col_res2:
            st.markdown("#### Umpan Balik Terukur")
            for msg in current["feedback"]:
                st.markdown(f"- {msg}")

            col_act1, col_act2 = st.columns(2)
            with col_act1:
                if st.button("Ulangi Latihan"):
                    clear_current_input_and_result()
                    st.rerun()
            with col_act2:
                st.page_link(
                    "pages/3_Bandingkan_Percobaan.py",
                    label="Buka Komparasi Percobaan →",
                )

        # Detail Teknis DSP
        with st.expander("Detail Pengukuran DSP & Karakteristik Fitur Sinyal"):
            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                st.markdown("**Statistik Frekuensi Dasar (Hz):**")
                st.write(f"- Mean F0: `{summary['mean_f0_hz']:.2f} Hz`")
                st.write(f"- Median F0: `{summary['median_f0_hz']:.2f} Hz`")
                st.write(f"- Simpangan Baku: `{summary['std_f0_hz']:.2f} Hz`")
                st.write(f"- Min – Max: `{summary['min_f0_hz']:.1f} Hz` – `{summary['max_f0_hz']:.1f} Hz`")
            with d_col2:
                st.markdown("**Kontur Relatif (Semitone):**")
                st.write(f"- Rentang Pitch: `{summary['pitch_range_semitones']:.2f} ST`")
                st.write(f"- Segmen Awal: `{summary['start_mean_st']:+.2f} ST`")
                st.write(f"- Segmen Akhir: `{summary['end_mean_st']:+.2f} ST`")
                st.write(f"- Kemiringan (Slope): `{summary['slope_semitones_per_sec']:+.2f} ST/detik`")
            with d_col3:
                st.markdown("**Kualitas Sinyal Asli:**")
                st.write(f"- Durasi Total: `{summary['duration_sec']:.2f} detik`")
                st.write(f"- Durasi Voiced: `{summary['voiced_duration_sec']:.2f} detik`")
                st.write(f"- Energi RMS: `{val_res.rms_energy:.4f}`")
                st.write(f"- Rasio Clipping: `{val_res.clipping_ratio*100:.2f}%`")

            fig_rel = create_pitch_contour_figure(
                timestamps=contour.timestamps,
                f0_values=current["relative_semitones"],
                y_label="Semitone Relatif terhadap Median",
                title="Kontur Pitch Relatif dalam Semitone (Normalisasi Median Pembicara)",
            )
            st.plotly_chart(fig_rel)

            fig_wave = create_waveform_figure(
                current["preprocessed"].samples,
                current["preprocessed"].sample_rate,
                time_offset_sec=current["preprocessed"].time_offset_sec,
            )
            st.plotly_chart(fig_wave)

        # 03. Komparasi Langsung Percobaan (Jika ada >= 2 percobaan pada target yang sama)
        matching_attempts = [
            a for a in st.session_state.get("attempts", [])
            if a.get("mode") == "SPEAKING" and a.get("target") == current["target"]
        ]
        if len(matching_attempts) >= 2:
            st.divider()
            st.markdown("### 03. Komparasi Langsung dengan Percobaan Sebelumnya")
            att_prev = matching_attempts[-2]
            att_curr = matching_attempts[-1]

            st.caption(
                f"Membandingkan **{att_prev['attempt_id']}** ({att_prev['timestamp']}) dengan "
                f"**{att_curr['attempt_id']}** ({att_curr['timestamp']}) pada pola target **{current['target']}**."
            )

            norm_axis = st.checkbox(
                "Normalisasi sumbu waktu (0% – 100%) untuk membandingkan bentuk kontur intonasi",
                value=True,
                key="inline_norm_time_speaking",
            )
            fig_cmp = create_comparison_overlay_figure(
                time_a=att_prev["contour"].timestamps,
                val_a=att_prev["relative_semitones"],
                label_a=f"{att_prev['attempt_id']} (Sebelumnya)",
                time_b=att_curr["contour"].timestamps,
                val_b=att_curr["relative_semitones"],
                label_b=f"{att_curr['attempt_id']} (Terbaru)",
                normalize_time=norm_axis,
                y_label="Kontur Relatif terhadap Median (Semitone)",
                title=f"Overlay Kontur Intonasi — {att_prev['attempt_id']} vs {att_curr['attempt_id']} (Target: {current['target']})",
            )
            st.plotly_chart(fig_cmp)

            sum_prev = att_prev.get("summary", {})
            sum_curr = att_curr.get("summary", {})
            cmp_df = pd.DataFrame(
                [
                    {
                        "Metrik Pengukuran": "Perubahan Pitch Akhir–Awal (ST)",
                        att_prev["attempt_id"]: f"{sum_prev.get('start_end_diff_semitones', 0.0):+.2f} ST",
                        att_curr["attempt_id"]: f"{sum_curr.get('start_end_diff_semitones', 0.0):+.2f} ST",
                    },
                    {
                        "Metrik Pengukuran": "Rentang Pitch Valid (ST)",
                        att_prev["attempt_id"]: f"{sum_prev.get('pitch_range_semitones', 0.0):.2f} ST",
                        att_curr["attempt_id"]: f"{sum_curr.get('pitch_range_semitones', 0.0):.2f} ST",
                    },
                    {
                        "Metrik Pengukuran": "Kemiringan Kontur (ST/detik)",
                        att_prev["attempt_id"]: f"{sum_prev.get('slope_semitones_per_sec', 0.0):+.2f}",
                        att_curr["attempt_id"]: f"{sum_curr.get('slope_semitones_per_sec', 0.0):+.2f}",
                    },
                    {
                        "Metrik Pengukuran": "Median F0 (Hz)",
                        att_prev["attempt_id"]: f"{sum_prev.get('median_f0_hz', 0.0):.1f} Hz ({sum_prev.get('median_note', '—')})",
                        att_curr["attempt_id"]: f"{sum_curr.get('median_f0_hz', 0.0):.1f} Hz ({sum_curr.get('median_note', '—')})",
                    },
                    {
                        "Metrik Pengukuran": "Cakupan Suara Voiced (%)",
                        att_prev["attempt_id"]: f"{sum_prev.get('voiced_ratio', 0.0)*100:.1f}% ({sum_prev.get('voiced_duration_sec', 0.0):.2f} dtk)",
                        att_curr["attempt_id"]: f"{sum_curr.get('voiced_ratio', 0.0)*100:.1f}% ({sum_curr.get('voiced_duration_sec', 0.0):.2f} dtk)",
                    },
                    {
                        "Metrik Pengukuran": "Hasil Klasifikasi SVM",
                        att_prev["attempt_id"]: att_prev.get("predicted_label") or "Belum tersedia",
                        att_curr["attempt_id"]: att_curr.get("predicted_label") or "Belum tersedia",
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
                "Tekan **Ulangi Latihan** atau **Rekam Ulang** dan rekam percobaan kedua pada target yang sama untuk melihat grafik overlay perbandingan langsung di halaman ini!"
            )
