"""Beranda Voice Analyzer — Platform Analisis Pitch dan Pola Intonasi Vokal."""

import numpy as np
import pandas as pd
import streamlit as st

from src.ui_components import init_session_state, inject_custom_css, reset_entire_session

st.set_page_config(
    page_title="Voice Analyzer — Audio Signal Processing & Pattern Recognition",
    page_icon="🎙️",
    layout="wide",
)

inject_custom_css()
init_session_state()

# Header Hero Section
st.markdown(
    """
    <div style="margin-bottom: 24px;">
      <span class="va-header-tag">Digital Signal Processing & Machine Learning</span>
      <h1 style="margin: 0; font-size: 2.3rem;">Voice Analyzer</h1>
      <p style="color: #64748b; font-size: 1.05rem; margin-top: 6px; max-width: 820px; line-height: 1.5;">
        Platform evaluasi kontur intonasi ujaran dan ketepatan nada bernyanyi berbasis estimasi frekuensi dasar (F0),
        pemrosesan sinyal digital, dan model klasifikasi Support Vector Machine (SVM).
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 3 Kartu Layanan Utama
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="va-card">
          <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #0284c7; letter-spacing: 0.05em; margin-bottom: 4px;">Modul 01</div>
          <h3 style="margin-top: 0; margin-bottom: 8px; color: #0f172a;">Latihan Intonasi Berbicara</h3>
          <p style="font-size: 0.88rem; color: #64748b; line-height: 1.5; margin-bottom: 16px;">
            Evaluasi pergerakan pitch frasa pendek (Naik, Turun, Datar, dsb.) dengan visualisasi semitone relatif dan klasifikasi otomatis pola prosodi menggunakan SVM.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Latihan_Berbicara.py", label="Buka Latihan Berbicara →")

with col2:
    st.markdown(
        """
        <div class="va-card">
          <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #059669; letter-spacing: 0.05em; margin-bottom: 4px;">Modul 02</div>
          <h3 style="margin-top: 0; margin-bottom: 8px; color: #0f172a;">Latihan Nada Bernyanyi</h3>
          <p style="font-size: 0.88rem; color: #64748b; line-height: 1.5; margin-bottom: 16px;">
            Dengarkan acuan nada lokal, ikuti nada target, dan ukur ketepatan vokal (selisih cents) serta kestabilan penahanan nada (variasi MAD) secara terpisah.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_Latihan_Bernyanyi.py", label="Buka Latihan Bernyanyi →")

with col3:
    attempts_cnt = len(st.session_state.get("attempts", []))
    st.markdown(
        f"""
        <div class="va-card">
          <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #d97706; letter-spacing: 0.05em; margin-bottom: 4px;">Modul 03</div>
          <h3 style="margin-top: 0; margin-bottom: 8px; color: #0f172a;">Komparasi Percobaan</h3>
          <p style="font-size: 0.88rem; color: #64748b; line-height: 1.5; margin-bottom: 16px;">
            Bandingkan overlay kontur pitch dua rekaman pada target yang sama dan ekspor rekapitulasi numerik ke format CSV/JSON. <em>(Tersimpan: {attempts_cnt} rekaman)</em>.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/3_Bandingkan_Percobaan.py", label="Buka Komparasi Percobaan →")

st.divider()

# Informasi Panduan & Prinsip Ilmiah
c_info1, c_info2 = st.columns(2)

with c_info1:
    st.markdown("### Panduan Rekaman")
    st.markdown(
        """
        - **Kondisi Akustik:** Rekam suara pada ruangan yang tenang dengan kebisingan latar minimal.
        - **Jarak Mikrofon:** Pertahankan jarak 10–20 cm dari mikrofon untuk mencegah energi terlalu rendah (*silence*) atau distorsi amplitudo (*clipping*).
        - **Unit Ujaran:** Untuk latihan berbicara, gunakan satu frasa pendek (2–5 kata) agar pola kontur terfokus.
        - **Nada Acuan:** Gunakan earphone/headphone saat mendengarkan nada acuan bernyanyi agar tidak bocor ke mikrofon.
        """
    )

with c_info2:
    st.markdown("### Prinsip Ilmiah & Batasan Sistem")
    st.markdown(
        """
        - **Pengukuran Komputasional:** Hasil analisis berakar pada ekstraksi F0 empiris (*librosa.pyin*) dan klasifikasi pola teruji (*Speaker-Independent SVM*).
        - **Bukan Diagnosis Medis:** Pengukuran ini dirancang sebagai instrumen latihan komputasional, bukan diagnosis kesehatan vokal, fisiologis, atau psikologis.
        - **Keamanan Data:** Pemrosesan audio dilakukan di server lokal/sesi Anda tanpa transmisi ke API generatif pihak ketiga.
        """
    )
    if st.button("Reset Seluruh Sesi"):
        reset_entire_session()
        st.toast("Sesi aktif berhasil dibersihkan.")
