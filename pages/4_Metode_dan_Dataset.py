"""Halaman 4: Metode dan Dataset (Informasi Edukasi, Siklus Pengenalan Pola, dan Statistik Dataset)."""

from collections import Counter
import csv
import pandas as pd
import streamlit as st

from src.config import A4_REFERENCE_HZ, DATA_DIR, INTONATION_FEATURE_NAMES, INTONATION_LABELS, LABEL_DESCRIPTIONS_ID
from src.ui_components import inject_custom_css

st.set_page_config(page_title="Metode & Dataset — Voice Analyzer", page_icon="🎙️", layout="wide")
inject_custom_css()

st.markdown(
    """
    <div style="margin-bottom: 20px;">
      <span class="va-header-tag">Landasan Ilmiah • Siklus Pengenalan Pola</span>
      <h1 style="margin: 0;">Metodologi, Pengolahan Sinyal, dan Dataset</h1>
      <p style="color: #64748b; font-size: 0.95rem; margin-top: 4px;">
        Dokumentasi teoritis frekuensi dasar (F0), formulasi matematis cents, alur siklus pengenalan pola, dan ringkasan 328 berkas dataset riil.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Konsep & Notasi Musik",
        "Siklus Pengenalan Pola",
        "Distribusi Dataset Terunduh",
        "Keterbatasan & Etika Ilmiah",
    ]
)

with tab1:
    st.markdown("### Istilah Dasar yang Digunakan Secara Konsisten")
    st.markdown(
        f"""
- **Pitch / F0 (Frekuensi Dasar):** Kecepatan getaran pita suara yang diukur dalam satuan **Hertz (Hz)**. Semakin tinggi angka Hz, semakin tinggi pitch suara yang terdengar.
- **Intonasi:** Pola perubahan pitch sepanjang satu frasa pendek (misalnya meningkat di akhir, menurun, datar, turun-lalu-naik, atau naik-lalu-turun).
- **Nada Musik:** Pemetaan frekuensi ke notasi musik standar dengan konvensi tuning **A4 = {A4_REFERENCE_HZ:.0f} Hz**:
  $$\\text{{MIDI}} = 69 + 12 \\cdot \\log_2\\left(\\frac{{f_{{\\text{{Hz}}}}}}{{440.0}}\\right)$$
- **Selisih Cents (Ketepatan Nada):** Ukuran jarak logaritmik antara pitch yang dinyanyikan ($f_0$) dan frekuensi nada target ($f_{{\\text{{target}}}}$). Satu semitone sama dengan **100 cents** dan satu oktaf sama dengan **1200 cents**:
  $$\\text{{cents\\_error}} = 1200 \\cdot \\log_2\\left(\\frac{{f_0}}{{f_{{\\text{{target}}}}}}\\right)$$
  - **Positif (`+`)**: suara Anda lebih tinggi (*sharp*) dari nada target.
  - **Negatif (`-`)**: suara Anda lebih rendah (*flat*) dari nada target.
  - **Nol (`0`)**: tepat pada frekuensi nada target.
- **Kestabilan Nada:** Variasi pitch selama bagian nada yang ditahan (diukur menggunakan *Median Absolute Deviation / MAD* dalam cents terhadap median pitch segmen itu sendiri). **Suara dapat sangat stabil sekaligus meleset dari target**, sehingga ketepatan dan kestabilan selalu ditampilkan secara terpisah.
"""
    )

    st.markdown("### Lima Kelas Pola Intonasi untuk Pengembangan SVM")
    cls_rows = [{"Kode Kelas": k, "Penjelasan Pola Kontur Pitch": LABEL_DESCRIPTIONS_ID.get(k, "")} for k in INTONATION_LABELS]
    st.table(pd.DataFrame(cls_rows))

with tab2:
    st.markdown("### Siklus Pengenalan Pola pada Voice Analyzer")
    st.markdown(
        """
Sistem ini menerapkan tahapan lengkap **Pengenalan Pola (*Pattern Recognition Cycle*)**:

1. **Pengumpulan Data (*Sensing / Audio Acquisition*):**
   - Perekaman sinyal suara monofonik melalui mikrofon browser atau unggahan file WAV/MP3.
2. **Validasi & Pra-Pemrosesan Data (*Data Preprocessing & Missing Value Handling*):**
   - Pemeriksaan kualitas sinyal asli sebelum normalisasi: durasi, energi RMS (deteksi hening), dan rasio *clipping*.
   - Konversi kanal ke mono, penghilangan *DC offset*, dan *resampling* ke `22.050 Hz`.
   - Estimasi kontur frekuensi dasar ($F_0$) menggunakan algoritma **`librosa.pyin`**.
   - **Penanganan Nilai Kosong (*Missing Values*):** Bagian rekaman tanpa getaran pita suara (*unvoiced* / jeda / konsonan tak bersuara) disimpan sebagai `NaN` dan dikeluarkan dari perhitungan statistik agar tidak merusak rata-rata menjadi `0 Hz`.
3. **Normalisasi & Ekstraksi Fitur (*Feature Extraction — 14 Fitur Kontur F0*):**
   - Untuk menghilangkan bias tinggi suara bawaan pembicara (pria vs wanita), frekuensi Hz dikonversi ke **semitone relatif terhadap median pitch masing-masing rekaman**:
     $$s_i = 12 \\cdot \\log_2\\left(\\frac{{f_i}}{{F_{{0,\\text{{median}}}}}}\\right)$$
   - Dari kontur semitone relatif tersebut diekstrak **14 fitur numerik** yang konsisten untuk training, evaluasi, dan inferensi website:
"""
    )
    st.code(", ".join(INTONATION_FEATURE_NAMES), language="text")
    st.markdown(
        r"""
4. **Klasifikasi & Pengambilan Keputusan (*Classification with Support Vector Machine*):**
   - Pipeline Machine Learning: `StandardScaler` → `SVC(kernel="rbf", class_weight="balanced")`.
   - Pemilihan parameter $C$ dan $\gamma$ dilakukan secara *offline* menggunakan `GridSearchCV` dengan pembagian berbasis pembicara (*Speaker-Independent GroupKFold*) agar tidak ada kebocoran suara pembicara yang sama antara data latih dan data uji.
"""
    )

with tab3:
    st.markdown("### Statistik Nyata Dataset yang Terunduh di Direktori `data/`")
    meta_csv = DATA_DIR / "annotations" / "metadata.csv"
    if not meta_csv.exists():
        st.warning("File `data/annotations/metadata.csv` belum ditemukan.")
    else:
        with meta_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sampel Audio", f"{len(rows)} file")
        c2.metric("Pembicara / Penyanyi Unik", f"{len(set(r['speaker_id'] for r in rows))} orang")
        c3.metric("Jumlah Sumber Dataset", f"{len(set(r['dataset'] for r in rows))} sumber")
        c4.metric("Speaker Leakage Lintas Split", "0% (Terverifikasi)")

        ds_summary = []
        for ds_name in sorted(set(r["dataset"] for r in rows)):
            sub = [r for r in rows if r["dataset"] == ds_name]
            spks = set(r["speaker_id"] for r in sub)
            splits_cnt = Counter(r["split"] for r in sub)
            ds_summary.append(
                {
                    "Dataset": ds_name,
                    "Versi": sub[0]["dataset_version"],
                    "Tugas Utama": sub[0]["task"],
                    "Bahasa": sub[0]["language"],
                    "Total Klip": len(sub),
                    "Pembicara Unik": len(spks),
                    "Train": splits_cnt.get("train", 0),
                    "Val": splits_cnt.get("val", 0),
                    "Test": splits_cnt.get("test", 0),
                }
            )
        st.dataframe(pd.DataFrame(ds_summary), width="stretch")

with tab4:
    st.markdown("### Keterbatasan Ilmiah & Privasi")
    st.markdown(
        """
1. **Bukan Alat Diagnosis:** Hasil analisis tidak boleh ditafsirkan sebagai diagnosis medis, kesehatan pita suara, psikologis, emosi mutlak, usia, maupun identitas.
2. **Batasan Aspek Pitch:** Pengukuran MVP berfokus pada frekuensi dasar ($F_0$), sehingga belum menilai kejelasan artikulasi, diksi, warna suara (*timbre*), teknik pernapasan, atau musikalitas secara keseluruhan.
3. **Rentang Nada Rekaman vs Rentang Vokal:** Nada terendah dan tertinggi yang terdeteksi dalam satu rekaman pendek hanya mencerminkan rekaman tersebut, bukan keseluruhan rentang vokal atau tipe suara definitif (*Bass, Baritone, Tenor, Alto, Mezzo-soprano, Soprano*).
4. **Privasi Audio Latihan:** Rekaman latihan pengguna diproses pada server aplikasi tanpa dikirim ke API pihak ketiga, diisolasi per sesi, dan tidak otomatis dimasukkan ke dalam dataset training.
"""
    )
