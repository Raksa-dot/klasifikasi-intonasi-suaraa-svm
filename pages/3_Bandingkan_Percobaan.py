"""Halaman 3: Bandingkan Percobaan dalam Sesi Latihan & Unduh Hasil (CSV/JSON)."""

import pandas as pd
import streamlit as st

from src.comparison import can_compare_attempts, summarize_attempt_comparison
from src.export import export_attempts_list_to_csv, export_summary_to_json
from src.ui_components import init_session_state, inject_custom_css, reset_entire_session
from src.visualization import create_comparison_overlay_figure

st.set_page_config(page_title="Komparasi Percobaan — Voice Analyzer", page_icon="🎙️", layout="wide")
inject_custom_css()
init_session_state()

st.markdown(
    """
    <div style="margin-bottom: 20px;">
      <span class="va-header-tag">Modul 03 • Analisis Komparatif</span>
      <h1 style="margin: 0;">Komparasi Percobaan Latihan</h1>
      <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px;">
        Bandingkan dua rekaman latihan pada target yang sama untuk mengevaluasi konsistensi dan progres intonasi/ketepatan nada Anda.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

attempts = st.session_state.get("attempts", [])

if not attempts:
    st.info(
        "Belum ada percobaan yang tersimpan di sesi ini. Silakan lakukan latihan di halaman **Latihan Berbicara** "
        "atau **Latihan Bernyanyi** terlebih dahulu."
    )
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        st.page_link("pages/1_Latihan_Berbicara.py", label="🗣️ Ke Latihan Berbicara →")
    with col_nav2:
        st.page_link("pages/2_Latihan_Bernyanyi.py", label="🎵 Ke Latihan Bernyanyi →")
else:
    # Tombol Unduh Seluruh Percobaan Sesi & Reset
    col_top1, col_top2, col_top3 = st.columns([2, 2, 1])
    with col_top1:
        csv_all = export_attempts_list_to_csv(attempts)
        st.download_button(
            "⬇️ Unduh Ringkasan Seluruh Percobaan Sesi (CSV)",
            data=csv_all.encode("utf-8"),
            file_name="voice_analyzer_percobaan_sesi.csv",
            mime="text/csv",
        )
    with col_top2:
        json_export_payload = {
            "total_attempts": len(attempts),
            "attempts": [
                {
                    "attempt_id": a.get("attempt_id"),
                    "timestamp": a.get("timestamp"),
                    "mode": a.get("mode"),
                    "exercise_id": a.get("exercise_id"),
                    "target": a.get("target"),
                    "phrase_text": a.get("phrase_text"),
                    "predicted_label": a.get("predicted_label"),
                    "summary": a.get("summary"),
                }
                for a in attempts
            ],
        }
        st.download_button(
            "⬇️ Unduh Detail Percobaan Sesi (JSON)",
            data=export_summary_to_json(json_export_payload).encode("utf-8"),
            file_name="voice_analyzer_percobaan_sesi.json",
            mime="application/json",
        )
    with col_top3:
        if st.button("🗑️ Reset Sesi"):
            reset_entire_session()
            st.rerun()

    st.divider()

    if len(attempts) < 2:
        st.warning(
            f"Saat ini baru tersimpan **1 percobaan** (`{attempts[0]['attempt_id']}` — Mode `{attempts[0]['mode']}`, "
            f"Target `{attempts[0]['target']}`). Lakukan minimal **1 percobaan lagi** pada latihan dan target yang sama "
            "untuk melihat grafik perbandingan berdampingan."
        )
    else:
        def _fmt_attempt(idx: int) -> str:
            a = attempts[idx]
            extra = f" | \"{a.get('phrase_text')}\"" if a.get("mode") == "SPEAKING" else ""
            return f"{a['attempt_id']} ({a['timestamp']}) — [{a['mode']}] Target: {a['target']}{extra}"

        col_pick1, col_pick2 = st.columns(2)
        with col_pick1:
            idx_a = st.selectbox("Pilih Percobaan A (Awal):", range(len(attempts)), index=max(0, len(attempts) - 2), format_func=_fmt_attempt)
        with col_pick2:
            idx_b = st.selectbox("Pilih Percobaan B (Pembanding):", range(len(attempts)), index=len(attempts) - 1, format_func=_fmt_attempt)

        att_a = attempts[idx_a]
        att_b = attempts[idx_b]

        # Pemutar Audio Masing-Masing Percobaan
        col_play1, col_play2 = st.columns(2)
        with col_play1:
            st.markdown(f"**🔊 Audio {att_a['attempt_id']}** (Target: `{att_a['target']}`)")
            if att_a.get("audio_wav_bytes"):
                st.audio(att_a["audio_wav_bytes"], format="audio/wav")
        with col_play2:
            st.markdown(f"**🔊 Audio {att_b['attempt_id']}** (Target: `{att_b['target']}`)")
            if att_b.get("audio_wav_bytes"):
                st.audio(att_b["audio_wav_bytes"], format="audio/wav")

        is_comparable, reason_msg = can_compare_attempts(att_a, att_b)
        if not is_comparable:
            st.warning(f"⚠️ **{reason_msg}**")
        else:
            st.success(f"✅ {reason_msg}")

            mode = att_a["mode"]
            sum_a = att_a.get("summary") or {}
            sum_b = att_b.get("summary") or {}

            if mode == "SPEAKING":
                norm_axis = st.checkbox(
                    "Normalisasi sumbu waktu menjadi Persentase Durasi (0% – 100%) untuk membandingkan bentuk kontur",
                    value=True,
                )
                st.caption(
                    "Catatan: Normalisasi waktu sederhana menyamakan skala durasi (0–100%), bukan membuktikan kesesuaian kata per kata. "
                    "Hilangkan centang di atas untuk melihat waktu asli (detik)."
                )
                fig_cmp = create_comparison_overlay_figure(
                    time_a=att_a["contour"].timestamps,
                    val_a=att_a["relative_semitones"],
                    label_a=f"{att_a['attempt_id']} ({att_a['timestamp']})",
                    time_b=att_b["contour"].timestamps,
                    val_b=att_b["relative_semitones"],
                    label_b=f"{att_b['attempt_id']} ({att_b['timestamp']})",
                    normalize_time=norm_axis,
                    y_label="Kontur Relatif terhadap Median (Semitone)",
                    title=f"Overlay Kontur Intonasi Relatif — Target: {att_a['target']} (\"{att_a.get('phrase_text', '')}\")",
                )
                st.plotly_chart(fig_cmp)

                cmp_df = pd.DataFrame(
                    [
                        {
                            "Metrik Pengukuran": "Perubahan Pitch Akhir–Awal (ST)",
                            att_a["attempt_id"]: f"{sum_a.get('start_end_diff_semitones', 0.0):+.2f} ST",
                            att_b["attempt_id"]: f"{sum_b.get('start_end_diff_semitones', 0.0):+.2f} ST",
                        },
                        {
                            "Metrik Pengukuran": "Rentang Pitch Valid (ST)",
                            att_a["attempt_id"]: f"{sum_a.get('pitch_range_semitones', 0.0):.2f} ST",
                            att_b["attempt_id"]: f"{sum_b.get('pitch_range_semitones', 0.0):.2f} ST",
                        },
                        {
                            "Metrik Pengukuran": "Kemiringan Kontur (ST/detik)",
                            att_a["attempt_id"]: f"{sum_a.get('slope_semitones_per_sec', 0.0):+.2f}",
                            att_b["attempt_id"]: f"{sum_b.get('slope_semitones_per_sec', 0.0):+.2f}",
                        },
                        {
                            "Metrik Pengukuran": "Median F0 (Hz)",
                            att_a["attempt_id"]: f"{sum_a.get('median_f0_hz', 0.0):.1f} Hz ({sum_a.get('median_note', '—')})",
                            att_b["attempt_id"]: f"{sum_b.get('median_f0_hz', 0.0):.1f} Hz ({sum_b.get('median_note', '—')})",
                        },
                        {
                            "Metrik Pengukuran": "Cakupan Suara Voiced (%)",
                            att_a["attempt_id"]: f"{sum_a.get('voiced_ratio', 0.0)*100:.1f}% ({sum_a.get('voiced_duration_sec', 0.0):.2f} dtk)",
                            att_b["attempt_id"]: f"{sum_b.get('voiced_ratio', 0.0)*100:.1f}% ({sum_b.get('voiced_duration_sec', 0.0):.2f} dtk)",
                        },
                        {
                            "Metrik Pengukuran": "Hasil Klasifikasi SVM",
                            att_a["attempt_id"]: att_a.get("predicted_label") or "Belum tersedia",
                            att_b["attempt_id"]: att_b.get("predicted_label") or "Belum tersedia",
                        },
                    ]
                )
                st.table(cmp_df)

            else:
                target_hz = float(att_a.get("target_hz", sum_a.get("target_hz", 220.0)))
                fig_cmp = create_comparison_overlay_figure(
                    time_a=att_a["contour"].timestamps,
                    val_a=att_a["contour"].f0_hz,
                    label_a=f"{att_a['attempt_id']} ({att_a['timestamp']})",
                    time_b=att_b["contour"].timestamps,
                    val_b=att_b["contour"].f0_hz,
                    label_b=f"{att_b['attempt_id']} ({att_b['timestamp']})",
                    normalize_time=False,
                    y_label="Frekuensi Dasar / F0 (Hz)",
                    title=f"Perbandingan Kontur Pitch terhadap Nada Target {att_a['target']} ({target_hz:.2f} Hz)",
                    target_line_y=target_hz,
                    target_line_label=f"Target {att_a['target']} ({target_hz:.2f} Hz)",
                )
                st.plotly_chart(fig_cmp)

                cmp_df = pd.DataFrame(
                    [
                        {
                            "Aspek Penilaian Terpisah": "Ketepatan: Median Selisih Bertanda (cents)",
                            att_a["attempt_id"]: f"{sum_a.get('median_signed_cents_error', 0.0):+.1f} cents",
                            att_b["attempt_id"]: f"{sum_b.get('median_signed_cents_error', 0.0):+.1f} cents",
                        },
                        {
                            "Aspek Penilaian Terpisah": "Ketepatan: Median Selisih Absolut (|cents|)",
                            att_a["attempt_id"]: f"{sum_a.get('median_abs_cents_error', 0.0):.1f} cents",
                            att_b["attempt_id"]: f"{sum_b.get('median_abs_cents_error', 0.0):.1f} cents",
                        },
                        {
                            "Aspek Penilaian Terpisah": "Kestabilan: Variasi MAD Segmen Inti (cents)",
                            att_a["attempt_id"]: f"{sum_a.get('stability_mad_cents', 0.0):.1f} cents",
                            att_b["attempt_id"]: f"{sum_b.get('stability_mad_cents', 0.0):.1f} cents",
                        },
                        {
                            "Aspek Penilaian Terpisah": f"Proporsi Frame dalam Toleransi (±{sum_a.get('tolerance_cents', 50.0):.0f} c)",
                            att_a["attempt_id"]: f"{sum_a.get('in_tolerance_ratio', 0.0)*100:.1f}%",
                            att_b["attempt_id"]: f"{sum_b.get('in_tolerance_ratio', 0.0)*100:.1f}%",
                        },
                        {
                            "Aspek Penilaian Terpisah": "Cakupan Data Valid (Voiced)",
                            att_a["attempt_id"]: f"{sum_a.get('voiced_duration_sec', 0.0):.2f} dtk ({sum_a.get('voiced_ratio', 0.0)*100:.1f}%)",
                            att_b["attempt_id"]: f"{sum_b.get('voiced_duration_sec', 0.0):.2f} dtk ({sum_b.get('voiced_ratio', 0.0)*100:.1f}%)",
                        },
                    ]
                )
                st.table(cmp_df)

            st.markdown("#### 📝 Ringkasan Perbandingan Objektif")
            for note in summarize_attempt_comparison(att_a, att_b):
                st.markdown(f"- {note}")
