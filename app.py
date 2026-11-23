"""Beranda Voice Analyzer — Website Latihan Berbicara dan Bernyanyi."""

import numpy as np
import pandas as pd
import streamlit as st

from src.ui_components import init_session_state, reset_entire_session

st.set_page_config(
    page_title="Voice Analyzer — Latihan Berbicara & Bernyanyi",
    page_icon="🎙️",
    layout="wide",
)

init_session_state()

st.title("🎙️ Voice Analyzer: Website Latihan Berbicara dan Bernyanyi")
st.markdown(
    """
**Voice Analyzer** adalah website latihan berbicara dan bernyanyi berbahasa Indonesia yang membantu Anda:
1. Memahami perubahan **pitch (frekuensi dasar / F0 dalam Hz)** sepanjang satu frasa pendek,
2. Mengikuti dan menahan **nada acuan musik** dengan pengukuran selisih **cents** dan **kestabilan nada**, serta
3. Membandingkan antarpercobaan latihan melalui umpan balik visual dan numerik yang terukur.
"""
)

st.markdown(
    """
<div style="background: #f0fdf4; border-left: 5px solid #16a34a; padding: 12px 16px; border-radius: 6px; margin-bottom: 18px;">
  <strong>Alur Utama Latihan:</strong><br/>
  Pilih latihan → Pahami target/contoh → Rekam atau unggah audio → Analisis → Lihat hasil → Coba lagi → Bandingkan percobaan
</div>
""",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🗣️ 1. Latihan Berbicara")
    st.write(
        "Latih pola intonasi pada **satu frasa pendek** (Naik, Turun, Datar, Turun-Lalu-Naik, Naik-Lalu-Turun). "
        "Dilengkapi visualisasi kontur pitch, perubahan semitone awal–akhir, monitor pitch live, dan klasifikasi pola intonasi."
    )
    st.page_link("pages/1_Latihan_Berbicara.py", label="Mulai Latihan Berbicara →", icon="🗣️")

with col2:
    st.markdown("### 🎵 2. Latihan Bernyanyi")
    st.write(
        "Dengarkan **nada acuan** lokal, ikuti dan tahan satu nada target, lalu ukur **ketepatan nada** (selisih bertanda dalam cents) "
        "serta **kestabilan nada** (variasi selama nada ditahan) secara terpisah."
    )
    st.page_link("pages/2_Latihan_Bernyanyi.py", label="Mulai Latihan Bernyanyi →", icon="🎵")

with col3:
    st.markdown("### 📊 3. Bandingkan Percobaan")
    attempts_cnt = len(st.session_state.get("attempts", []))
    st.write(
        f"Bandingkan dua percobaan dari mode, latihan, dan target yang sama dalam sesi ini, "
        f"serta unduh hasil latihan ke format CSV/JSON. *(Tersimpan saat ini: **{attempts_cnt} percobaan**)*."
    )
    st.page_link("pages/3_Bandingkan_Percobaan.py", label="Buka Perbandingan Percobaan →", icon="📊")

st.divider()

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.markdown("#### 📌 Petunjuk Merekam yang Baik")
    st.markdown(
        """
- Gunakan tempat yang **cukup tenang** dan minim gema atau kebisingan latar.
- Jaga jarak mikrofon sekitar **10–20 cm** dari mulut agar suara tidak terlalu pelan (*hening*) dan tidak pecah (*clipping*).
- Pada latihan bernyanyi, gunakan **headphone/earphone** saat mendengarkan nada acuan, dan pastikan nada acuan sudah berhenti sebelum mulai merekam.
- Satu rekaman latihan berbicara diperuntukkan bagi **satu frasa pendek**.
"""
    )

with col_info2:
    st.markdown("#### ⚖️ Prinsip Ilmiah & Batasan Hasil")
    st.markdown(
        """
- Hasil yang ditampilkan adalah **pengukuran komputasional** terhadap frekuensi dasar (*pitch/F0*) pada latihan yang dipilih.
- Sistem ini **bukan alat diagnosis** kondisi kesehatan, psikologis, usia, atau identitas, dan belum menilai keseluruhan aspek artikulasi, ekspresi, atau musikalitas vokal.
- Audio latihan diproses di server aplikasi tanpa dikirim ke layanan pihak ketiga dan hanya disimpan sementara selama sesi latihan aktif.
"""
    )
    if st.button("🗑️ Bersihkan & Reset Sesi Latihan Saat Ini"):
        reset_entire_session()
        st.success("Sesi latihan telah direset.")
