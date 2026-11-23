"""Halaman 1: Latihan Berbicara (Pola Intonasi Satu Frasa Pendek)."""

from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

from src.audio_io import encode_wav_bytes, read_audio_bytes
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
    render_live_pitch_monitor,
    render_shared_audio_input,
    save_attempt_to_session,
)
from src.validation import validate_raw_signal
from src.visualization import (
    create_pitch_contour_figure,
    create_target_guide_figure,
    create_waveform_figure,
)

st.set_page_config(page_title="Latihan Berbicara — Voice Analyzer", page_icon="🗣️", layout="wide")
init_session_state()

st.title("🗣️ Latihan Berbicara — Pola Intonasi Satu Frasa Pendek")
st.write(
    "Unit latihan dan klasifikasi adalah **satu frasa pendek**. Pilih pola target intonasi dan frasa latihan di bawah ini, "
    "pahami panduan konturnya, lalu rekam atau unggah suara Anda."
)

exercises_list = load_exercises().get("speaking_exercises", [])

col_sel1, col_sel2 = st.columns([1, 1])
with col_sel1:
    source_mode = st.radio(
        "Sumber Frasa Latihan:",
        ["Pilih dari Daftar Frasa Latihan", "Masukkan Frasa Pendek Sendiri"],
        horizontal=True,
    )

if source_mode == "Pilih dari Daftar Frasa Latihan" and exercises_list:
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
            "Pilih Target Pola Intonasi:",
            list(INTONATION_LABELS),
            format_func=lambda k: f"{k} — {LABEL_DESCRIPTIONS_ID.get(k, k)}",
        )
    custom_phrase = st.text_input(
        "Masukkan Teks Satu Frasa Pendek (maks. 60 karakter):",
        value="Sudah siap mulai sekarang?",
        max_chars=60,
    ).strip()
    phrase_text = custom_phrase or "Sudah siap mulai?"
    exercise_id = f"custom_spk_{target_label.lower()}"
    instruction_text = f"Ucapkan satu frasa pendek \"{phrase_text}\" dengan mengikuti target {LABEL_DESCRIPTIONS_ID.get(target_label, target_label)}."

# Panel Target & Ilustrasi Panduan
st.markdown("### 🎯 1. Target Latihan & Panduan Kontur")
col_t1, col_t2 = st.columns([1, 1])
with col_t1:
    st.info(
        f"**Target Pola Intonasi:** `{target_label}` ({LABEL_DESCRIPTIONS_ID.get(target_label, '')})\n\n"
        f"**Frasa yang Diucapkan:** *\"{phrase_text}\"*\n\n"
        f"**Instruksi Latihan:** {instruction_text}"
    )
    st.caption(
        "Catatan: Memilih target latihan di atas adalah sasaran latihan Anda; hasil pengukuran aktual dan klasifikasi dihitung secara objektif dari rekaman suara Anda."
    )
with col_t2:
    st.plotly_chart(create_target_guide_figure(target_label))

# Monitor Pitch Live Real-Time (Opsional di Browser)
with st.expander("⚡ Buka Monitor Pitch & Tren Kontur Real-Time (Live saat Bersuara di Mikrofon)", expanded=False):
    render_live_pitch_monitor(mode="SPEAKING", target_hz=None, target_label=target_label)

st.divider()

# Input Audio Bersama
audio_bytes, filename_hint, analyze_clicked = render_shared_audio_input(
    mode="SPEAKING",
    exercise_id=exercise_id,
    target_key=f"{target_label}:{phrase_text}",
)

if analyze_clicked and audio_bytes is not None:
    with st.spinner("Memvalidasi sinyal audio, mengekstraksi pitch F0 (librosa.pyin), dan menganalisis kontur..."):
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
                "prediction": pred_res,
                "predicted_label": pred_res.predicted_label if pred_res.is_available else None,
                "feedback": feedback_msgs,
                "audio_wav_bytes": wav_play_bytes,
            }
            st.session_state["current_analysis"] = analysis_payload

            if val_res.is_valid and summary is not None:
                att_num = save_attempt_to_session(analysis_payload)
                st.toast(f"Hasil analisis disimpan sebagai Percobaan #{att_num} di sesi ini.")
        except Exception as exc:
            st.error(f"Gagal memproses file audio. Pastikan file berformat WAV/MP3 yang valid ({exc}).")

# Tampilkan Hasil Analisis
current = st.session_state.get("current_analysis")
if current and current.get("mode") == "SPEAKING":
    st.divider()
    st.markdown("### 📈 2. Hasil Analisis Latihan Berbicara")

    val_res = current["validation"]
    summary = current["summary"]
    contour = current["contour"]
    pred_res = current["prediction"]

    if val_res.warnings:
        for w in val_res.warnings:
            st.warning(w)

    if not val_res.is_valid or summary is None:
        st.error("Rekaman belum memenuhi syarat untuk penilaian kontur pitch yang valid:")
        for reason in val_res.reasons:
            st.markdown(f"- {reason}")
        if summary is None:
            voiced_cnt = int(contour.voiced_mask.sum()) if contour.f0_hz.size > 0 else 0
            st.markdown(
                f"- Bagian suara yang memiliki pitch valid (*voiced frames*) terlalu sedikit (**{voiced_cnt} frame**). "
                "Silakan rekam ulang dengan suara yang lebih jelas di tempat yang tenang."
            )
    else:
        # Baris Metrik Utama (Memisahkan Target, Pengukuran Aktual, Kualitas Sinyal, dan Hasil Klasifikasi)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Target Latihan", f"{current['target']}")
        m2.metric(
            "Perubahan Pitch Akhir–Awal",
            f"{summary['start_end_diff_semitones']:+.2f} ST",
            help="Selisih rata-rata pitch segmen akhir dikurangi segmen awal dalam semitone.",
        )
        m3.metric(
            "Median Pitch / F0",
            f"{summary['median_f0_hz']:.1f} Hz ({summary['median_note']})",
            help="Frekuensi dasar median pada bagian suara yang bernada (voiced).",
        )
        m4.metric(
            "Cakupan Suara Voiced",
            f"{summary['voiced_ratio'] * 100:.1f}%",
            help=f"{summary['voiced_frames']} dari {summary['total_frames']} frame memiliki pitch valid.",
        )

        # Grafik Kontur Pitch terhadap Waktu (Hz)
        fig_f0 = create_pitch_contour_figure(
            timestamps=contour.timestamps,
            f0_values=contour.f0_hz,
            y_label="Frekuensi Dasar / F0 (Hz)",
            title=f"Kontur Pitch (F0) terhadap Waktu — Frasa: \"{current['phrase_text']}\"",
        )
        st.plotly_chart(fig_f0)

        # Bagian Hasil Klasifikasi SVM vs Pengukuran Aktual
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown("#### 🤖 Hasil Klasifikasi Pola Intonasi (SVM)")
            if pred_res.is_available and pred_res.predicted_label:
                matches = pred_res.predicted_label == current["target"]
                badge_color = "#16a34a" if matches else "#d97706"
                match_text = "Sesuai dengan Target Latihan" if matches else f"Berbeda dari Target ({current['target']})"
                st.markdown(
                    f"""
                    <div style="border: 1px solid {badge_color}; background: #ffffff; padding: 14px; border-radius: 8px;">
                      <div style="font-size: 12px; color: #64748b;">Pola Intonasi Terprediksi (Model {pred_res.model_version}):</div>
                      <div style="font-size: 22px; font-weight: 700; color: {badge_color}; margin: 4px 0;">{pred_res.predicted_label}</div>
                      <div style="font-size: 13px; color: #334155;">{LABEL_DESCRIPTIONS_ID.get(pred_res.predicted_label, '')}</div>
                      <div style="font-size: 12px; font-weight: 600; color: {badge_color}; margin-top: 6px;">Status: {match_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info(pred_res.status_message)

        with col_res2:
            st.markdown("#### 💡 Saran Latihan & Umpan Balik Terukur")
            for msg in current["feedback"]:
                st.markdown(f"- {msg}")

            col_act1, col_act2 = st.columns(2)
            with col_act1:
                if st.button("🎙️ Coba Lagi Latihan Ini"):
                    clear_current_input_and_result()
                    st.rerun()
            with col_act2:
                st.page_link(
                    "pages/3_Bandingkan_Percobaan.py",
                    label="📊 Bandingkan Percobaan →",
                )

        # Detail Teknis yang Dapat Dilipat (Expander)
        with st.expander("🔬 Buka Detail Teknis (Statistik F0, Kontur Semitone Relatif, Waveform & Kualitas Sinyal)"):
            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                st.markdown("**Statistik Frekuensi Dasar (Hz):**")
                st.write(f"- Mean F0: `{summary['mean_f0_hz']:.2f} Hz`")
                st.write(f"- Median F0: `{summary['median_f0_hz']:.2f} Hz`")
                st.write(f"- Simpangan Baku (Std): `{summary['std_f0_hz']:.2f} Hz`")
                st.write(f"- Min – Max Valid: `{summary['min_f0_hz']:.2f} Hz` ({summary['lowest_note']}) – `{summary['max_f0_hz']:.2f} Hz` ({summary['highest_note']})")
                st.write(f"- Persentil P10 – P90: `{summary['p10_f0_hz']:.2f} Hz` – `{summary['p90_f0_hz']:.2f} Hz`")
            with d_col2:
                st.markdown("**Kontur Relatif (Semitone terhadap Median):**")
                st.write(f"- Rentang Pitch: `{summary['pitch_range_semitones']:.2f} ST` (`{summary['pitch_range_hz']:.2f} Hz`)")
                st.write(f"- Mean Segmen Awal: `{summary['start_mean_st']:+.2f} ST`")
                st.write(f"- Mean Segmen Tengah: `{summary['mid_mean_st']:+.2f} ST`")
                st.write(f"- Mean Segmen Akhir: `{summary['end_mean_st']:+.2f} ST`")
                st.write(f"- Kemiringan (Slope): `{summary['slope_semitones_per_sec']:+.2f} ST/detik`")
            with d_col3:
                st.markdown("**Durasi & Kualitas Sinyal:**")
                st.write(f"- Durasi Total Rekaman: `{summary['duration_sec']:.2f} detik`")
                st.write(f"- Durasi Bagian Voiced: `{summary['voiced_duration_sec']:.2f} detik`")
                st.write(f"- Proporsi Frame Voiced: `{summary['voiced_frames']} / {summary['total_frames']}` (`{summary['voiced_ratio']*100:.1f}%`)")
                st.write(f"- Energi RMS Sinyal Asli: `{val_res.rms_energy:.4f}`")
                st.write(f"- Rasio Clipping: `{val_res.clipping_ratio*100:.2f}%`")

            fig_rel = create_pitch_contour_figure(
                timestamps=contour.timestamps,
                f0_values=current["relative_semitones"],
                y_label="Pitch Relatif terhadap Median (Semitone)",
                title="Kontur Pitch Relatif dalam Semitone (Digunakan untuk Perbandingan Bentuk Intonasi)",
            )
            st.plotly_chart(fig_rel)

            fig_wave = create_waveform_figure(
                current["preprocessed"].samples,
                current["preprocessed"].sample_rate,
                time_offset_sec=current["preprocessed"].time_offset_sec,
            )
            st.plotly_chart(fig_wave)
