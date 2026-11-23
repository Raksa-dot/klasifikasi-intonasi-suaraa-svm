# TASK.md — Daftar Tiket Pengerjaan Proyek (Project Tickets)

> Dokumen ini berisi **tiket pengerjaan proyek (`TICKET-01` s.d. `TICKET-12`)** yang mengintegrasikan **Milestone pada [`AGENTS.md`](./AGENTS.md)** dengan **5 Komponen Penilaian Tugas Kuliah *Project Pengenalan Pola* (masing-masing 20%)**.

**Legenda Status:**
- `✅ SELESAI` — Sudah dikerjakan dan terverifikasi di workspace.
- `⏳ HARUS DIKERJAKAN` — Tugas berikutnya yang perlu diimplementasikan secara berurutan.

---

## Ringkasan Papan Tiket (Ticket Board)

| ID Tiket | Fase | Judul Tugas Pengerjaan | Prioritas | Keterkaitan Tugas Kuliah (PDF) & Milestone | Status |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **`TICKET-01`** | Fase 1 | Spesifikasi `AGENTS.md`, `TASK.md`, dan Struktur Folder Proyek | `P0` | Milestone 1 (Kerangka Proyek) | `✅ SELESAI` |
| **`TICKET-02`** | Fase 1 | Pengunduhan 4 Dataset Prioritas (328 Audio + 132 Anotasi) & `metadata.csv` | `P0` | Milestone 6 & Bab 3 PDF (Dataset) | `✅ SELESAI` |
| **`TICKET-03`** | Fase 2 | Instalasi Environment & Pipeline DSP (Validasi Audio, Preprocessing, Ekstraksi $F_0$ `pyin`, Nada & Cents) | `P0` | Milestone 2–3 & Bab 2–3 PDF (Preprocessing) | `✅ SELESAI` |
| **`TICKET-04`** | Fase 3 | Anotasi & Verifikasi Label 5 Kelas Intonasi Ujaran Bahasa Indonesia + *Speaker-Independent Split* | `P0` | Milestone 6 & Bab 3 PDF (Karakteristik Data) | `✅ SELESAI` |
| **`TICKET-05`** | Fase 3 | Ekstraksi Vektor Fitur Numerik $F_0$ Dataset ke `data/features/intonation_features.csv` | `P0` | Milestone 7 & Bab 3 PDF (Feature Extraction) | `✅ SELESAI` |
| **`TICKET-06`** | Fase 3 | Pelatihan Model SVM (`GridSearchCV`), Skenario Eksperimen & Evaluasi *Confusion Matrix* | `P0` | Milestone 7 & Bab 3–4 PDF (Training & Evaluasi 40%) | `✅ SELESAI` |
| **`TICKET-07`** | Fase 4 | Komponen Input Audio Bersama (Rekam Mikrofon Browser, Unggah WAV, Player & *Live Pitch Monitor*) | `P1` | Milestone 2 (Input & Pemutar Audio) | `✅ SELESAI` |
| **`TICKET-08`** | Fase 4 | Dashboard Halaman `1_Latihan_Berbicara.py` (Kontur $F_0$, Statistik Semitone, Inferensi SVM & Feedback) | `P1` | Milestone 4 & 7 (Website Latihan Berbicara) | `✅ SELESAI` |
| **`TICKET-09`** | Fase 4 | Dashboard Halaman `2_Latihan_Bernyanyi.py` (Nada Acuan, Ketepatan *Cents*, Kestabilan MAD & Grafik Target) | `P1` | Milestone 4 (Website Latihan Bernyanyi) | `✅ SELESAI` |
| **`TICKET-10`** | Fase 4 | Dashboard Halaman `3_Bandingkan_Percobaan.py` (*Overlay* Percobaan 1 vs 2 & Unduh CSV/JSON) | `P1` | Milestone 4 (Perbandingan Percobaan) | `✅ SELESAI` |
| **`TICKET-11`** | Fase 4 | Dashboard Halaman `4_Metode_dan_Dataset.py` & `5_Evaluasi_Model.py` (Visualisasi *Confusion Matrix* & Metrik Nyata) | `P1` | Milestone 5 & 7, Bab 2–4 PDF (Dashboard Evaluasi) | `✅ SELESAI` |
| **`TICKET-12`** | Fase 5 | Pengujian Unit (`pytest`), Uji End-to-End Streamlit & Penyusunan Laporan 5 Bab Tugas Pengenalan Pola | `P2` | Milestone 5 & Bab 1–5 PDF (100% Rubrik Tugas) | `✅ SELESAI` |

---

## Rincian Detail Setiap Tiket Pengerjaan

### FASE 1 — Fondasi Proyek & Akuisisi Dataset *(Sudah Selesai)*

#### `TICKET-01` — Spesifikasi `AGENTS.md`, `TASK.md`, dan Struktur Folder Proyek (`✅ SELESAI`)
- **File Terkait:** `AGENTS.md`, `TASK.md`, `README.md`, `requirements.txt`, `.gitignore`, serta direktori `pages/`, `src/`, `resources/`, `training/`, `models/`, `data/`, `reports/`, `tests/`.
- **Hasil:** Seluruh aturan ilmiah, struktur modul Python/Streamlit, dan daftar latihan awal di `resources/exercises.json` telah terbentuk.

#### `TICKET-02` — Pengunduhan 4 Dataset Prioritas & Penyusunan `metadata.csv` (`✅ SELESAI`)
- **File Terkait:** `training/download_datasets.py`, `data/annotations/metadata.csv`, `data/README.md`.
- **Hasil:**
  - Terunduh **328 sampel audio asli** (**91,45 MB**) dari **46 pembicara/penyanyi unik** beserta **132 file anotasi/referensi**:
    1. *Common Voice Scripted Speech Bahasa Indonesia (`v22.0`)*: 220 MP3 (15 pembicara) + `test.tsv` & `dev.tsv`.
    2. *PTDB-TUG*: 50 WAV (14 pembicara) + 50 CSV referensi $F_0$ laryngograph.
    3. *VocalSet v1.2 + Annotated-VocalSet*: 38 WAV (14 penyanyi) + 40 CSV anotasi $F_0$/nada.
    4. *GTSinger*: 20 WAV (10 nyanyian + 10 *paired speech*, 3 penyanyi) + 20 JSON + 20 TextGrid.
  - Pembagian *Speaker-Independent Split* (`train`: 186, `val`: 72, `test`: 70) bebas kebocoran pembicara.

---

### FASE 2 — Inti Pengolahan Sinyal Audio & Pengukuran $F_0$ (`✅ SELESAI`)

#### `TICKET-03` — Instalasi Environment & Pipeline Pengolahan Sinyal Audio (`✅ SELESAI`)
- **Prioritas:** `P0 (Wajib Utama)` — **Urutan Eksekusi #1**
- **File yang Dikerjakan:**
  - `src/config.py`, `src/audio_io.py`, `src/validation.py`, `src/preprocessing.py`, `src/pitch.py`, `src/notes.py`, `src/features.py`
- **Daftar Tugas:**
  - [x] Menginstal pustaka pada `requirements.txt` (`streamlit`, `librosa`, `soundfile`, `numpy`, `scipy`, `pandas`, `plotly`, `scikit-learn`, `joblib`, `pytest`).
  - [x] Memastikan pembacaan audio WAV & MP3 berjalan lancar serta pemeriksaan kualitas sinyal asli (durasi, energi RMS untuk deteksi hening, rasio *clipping*) dilakukan sebelum normalisasi.
  - [x] Mengimplementasikan *preprocessing* (konversi mono, hapus DC offset, resample ke `22.050 Hz`).
  - [x] Mengimplementasikan ekstraksi kontur $F_0$ dengan `librosa.pyin`, memisahkan frame *voiced* vs *unvoiced* (penanganan *missing values* `NaN` tanpa memalsukan pitch `0 Hz`).
  - [x] Memverifikasi akurasi ekstraksi $F_0$ dan konversi nada/cents menggunakan sampel referensi **PTDB-TUG** dan **Annotated-VocalSet**.
- **Kriteria Selesai (DoD):** Ekstraksi $F_0$ menghasilkan kontur, timestamp, rasio *voiced*, nama nada (`A4 = 440 Hz`), dan selisih *cents* yang akurat serta lolos uji pada sinyal referensi.

---

### FASE 3 — Pipeline Machine Learning SVM *(Inti Penilaian Bab 3 & 4 Tugas Kuliah)* (`✅ SELESAI`)

#### `TICKET-04` — Anotasi & Verifikasi Label 5 Kelas Intonasi Ujaran + *Speaker-Independent Split* (`✅ SELESAI`)
- **Prioritas:** `P0 (Wajib Utama)` — **Urutan Eksekusi #2**
- **File yang Dikerjakan:**
  - `training/split_dataset.py`, `data/annotations/metadata.csv`, `data/annotations/intonation_dataset.csv`
- **Daftar Tugas:**
  - [x] Mengekstrak kontur $F_0$ pada sampel ujaran (*Common Voice Bahasa Indonesia* & *paired speech*) dan menjalankan prosedur penetapan serta verifikasi label 5 kelas intonasi: `RISING`, `FALLING`, `LEVEL`, `FALL_RISE`, `RISE_FALL` (menyisihkan sampel yang rusak atau terlalu pendek/ambigu).
  - [x] Mencatat sumber label secara jujur sesuai aturan Bagian 4.4 `AGENTS.md` (memisahkan `target_label` dari `intonation_label` dan `annotation_source`).
  - [x] Memastikan pembagian `train`, `val`, dan `test` dilakukan secara *speaker-independent* (`speaker_id` yang sama tidak pernah bocor antar-split) dan melaporkan distribusi jumlah sampel serta pembicara per kelas.
- **Kriteria Selesai (DoD):** Tersedia tabel dataset intonasi berlabel valid lintas 5 kelas dengan pemisahan pembicara yang terverifikasi 0% *leakage*.

#### `TICKET-05` — Ekstraksi Vektor Fitur Numerik $F_0$ Dataset (`✅ SELESAI`)
- **Prioritas:** `P0 (Wajib Utama)` — **Urutan Eksekusi #3**
- **File yang Dikerjakan:**
  - `src/features.py`, `training/extract_dataset_features.py`, `data/features/intonation_features.csv`
- **Daftar Tugas:**
  - [x] Menyatukan fungsi ekstraksi vektor fitur intonasi di `src/features.py` agar urutan dan perhitungannya identik untuk training, evaluasi, dan inferensi website.
  - [x] Mengekstrak fitur kontur $F_0$ (statistik Hz & normalisasi semitone relatif per pembicara: mean, median, std, min, max, persentil P10/P90, rentang semitone, slope keseluruhan, slope segmen awal/tengah/akhir, selisih awal–tengah–akhir, koefisien kelengkungan kurva, dan proporsi *voiced*).
  - [x] Menyimpan tabel fitur lengkap beserta `speaker_id`, `split`, dan `intonation_label` ke `data/features/intonation_features.csv`.
- **Kriteria Selesai (DoD):** Skrip `python -m training.extract_dataset_features` berhasil menghasilkan `data/features/intonation_features.csv` tanpa nilai `NaN`/`Inf` pada vektor fitur yang valid.

#### `TICKET-06` — Pelatihan Model SVM, Skenario Eksperimen & Evaluasi *Confusion Matrix* (`✅ SELESAI`)
- **Prioritas:** `P0 (Wajib Utama)` — **Urutan Eksekusi #4** *(Mencakup 40% bobot nilai PDF Tugas Pengenalan Pola: Bab 3 & Bab 4)*
- **File yang Dikerjakan:**
  - `training/train_intonation.py`, `training/evaluate.py`, `models/intonation_svm.joblib`, `models/intonation_svm_metadata.json`, `reports/evaluation_report.json`, `reports/experiment_scenarios.csv`
- **Daftar Tugas:**
  - [x] Melatih model *baseline* pembanding dan pipeline utama `StandardScaler -> SVC(class_weight="balanced")`.
  - [x] Menjalankan **Skenario Eksperimen** sesuai rubrik PDF Tugas Pengenalan Pola:
    1. Variasi kernel SVM (`rbf` vs `linear` / `poly`) dan *hyperparameter* $C$ serta $\gamma$ menggunakan `GridSearchCV` + `GroupKFold` / `StratifiedGroupKFold` berbasis `speaker_id`.
    2. Skenario perbandingan tanpa reduksi dimensi vs dengan reduksi dimensi (`PCA`).
  - [x] Mengevaluasi model pada *speaker-independent test set* yang terpisah:
    - Menghitung **Accuracy, Balanced Accuracy, Macro-F1, Precision, Recall, dan F1-Score per kelas**.
    - Membentuk matriks **Confusion Matrix** 5 kelas.
    - Membandingkan skor *Training / Cross-Validation* vs *Testing* untuk analisis **Overfitting / Underfitting** dan analisis kesalahan klasifikasi (*Misclassification*).
  - [x] Menyimpan artefak model `models/intonation_svm.joblib`, metadata lengkap `models/intonation_svm_metadata.json`, serta laporan evaluasi `reports/evaluation_report.json`.
- **Kriteria Selesai (DoD):** Model SVM terlatih dan tervalidasi tersimpan di `models/` lengkap dengan laporan metrik nyata di `reports/`.

---

### FASE 4 — Pembangunan Dashboard Website Interaktif (Streamlit) (`✅ SELESAI`)

#### `TICKET-07` — Komponen Input Audio Bersama (Mikrofon Browser, Unggah WAV & *Live Pitch Monitor*) (`✅ SELESAI`)
- **Prioritas:** `P1 (Wajib Website)` — **Urutan Eksekusi #5**
- **File yang Dikerjakan:**
  - `src/audio_io.py`, `src/visualization.py`
- **Daftar Tugas:**
  - [x] Menyediakan perekam mikrofon langsung di browser (`st.audio_input`) dan unggah file WAV (serta dukungan MP3).
  - [x] Menambahkan komponen opsional **Live Pitch Monitor / Tuner Real-Time** berbasis Web Audio API di browser agar pengguna dapat melihat respons frekuensi (Hz) & nada secara langsung saat berbicara/bernyanyi di mikrofon.
  - [x] Menampilkan pemutar audio (*audio player*), informasi durasi, grafik *waveform*, status proses, tombol **Rekam Ulang**, dan tombol **Reset Sesi**.
  - [x] Memastikan setiap perubahan file audio, mode, atau target latihan otomatis menghapus hasil analisis lama di `st.session_state`.
- **Kriteria Selesai (DoD):** Pengguna dapat merekam dari mikrofon atau mengunggah file audio, memutarnya kembali, dan mereset sesi dengan bersih.

#### `TICKET-08` — Dashboard Halaman `1_Latihan_Berbicara.py` (`✅ SELESAI`)
- **Prioritas:** `P1 (Wajib Website)` — **Urutan Eksekusi #6**
- **File yang Dikerjakan:**
  - `pages/1_Latihan_Berbicara.py`, `src/prediction.py`, `src/feedback.py`, `src/visualization.py`
- **Daftar Tugas:**
  - [x] Menampilkan pilihan pola target (`RISING`, `FALLING`, `LEVEL`, `FALL_RISE`, `RISE_FALL`), pilihan frasa pendek dari `resources/exercises.json` atau input teks frasa pendek sendiri.
  - [x] Menampilkan instruksi latihan dan grafik **Ilustrasi Panduan Pola Target** (diberi label jelas sebagai panduan).
  - [x] Menampilkan hasil analisis: kartu status kualitas sinyal, grafik interaktif **Kontur Pitch ($F_0$) terhadap Waktu**, ringkasan perubahan pitch awal–akhir (dalam semitone), **Hasil Prediksi Pola Intonasi SVM** (beserta status kesesuaian dengan target), dan saran latihan.
  - [x] Menyediakan bagian detail teknis yang dapat dilipat (*expander*: mean/median F0, min/max valid, persentil, pitch range, durasi, waveform, proporsi *voiced*, dan grafik kontur relatif semitone).
  - [x] Menyimpan hasil percobaan ke dalam state sesi agar dapat dibandingkan di halaman perbandingan.
- **Kriteria Selesai (DoD):** Halaman Latihan Berbicara berjalan penuh dari pemilihan target, rekam/unggah audio, visualisasi kontur, prediksi SVM, hingga simpan percobaan.

#### `TICKET-09` — Dashboard Halaman `2_Latihan_Bernyanyi.py` (`✅ SELESAI`)
- **Prioritas:** `P1 (Wajib Website)` — **Urutan Eksekusi #7**
- **File yang Dikerjakan:**
  - `pages/2_Latihan_Bernyanyi.py`, `src/reference_tones.py`, `src/notes.py`, `src/feedback.py`, `src/visualization.py`
- **Daftar Tugas:**
  - [x] Menampilkan pilihan nada target (`A2` s.d. `A4` dengan tuning `A4 = 440 Hz`) dan pemutar suara **Nada Acuan** lokal.
  - [x] Menampilkan grafik kontur pitch pengguna bersama **garis horizontal nada target** dan pita zona toleransi (`±50 cents`).
  - [x] Menampilkan kartu metrik terpisah untuk:
    1. **Nada Target vs Nada Terdeteksi** (tanpa mengabaikan selisih oktaf 1200 cents),
    2. **Ketepatan Nada** (median selisih bertanda dalam *cents*: `+` lebih tinggi / *sharp*, `-` lebih rendah / *flat*, serta median selisih absolut),
    3. **Kestabilan Nada** (variasi MAD dalam *cents* pada bagian stabil),
    4. **Cakupan Data Valid** (durasi *voiced*, proporsi *voiced*, dan persentase frame dalam toleransi).
  - [x] Menampilkan saran latihan bernyanyi berbasis pengukuran dan menyimpan percobaan ke sesi.
- **Kriteria Selesai (DoD):** Halaman Latihan Bernyanyi dapat memutar nada acuan, menganalisis rekaman nyanyian, serta menampilkan ketepatan dan kestabilan secara terpisah.

#### `TICKET-10` — Dashboard Halaman `3_Bandingkan_Percobaan.py` & Ekspor CSV/JSON (`✅ SELESAI`)
- **Prioritas:** `P1 (Wajib Website)` — **Urutan Eksekusi #8**
- **File yang Dikerjakan:**
  - `pages/3_Bandingkan_Percobaan.py`, `src/comparison.py`, `src/export.py`
- **Daftar Tugas:**
  - [x] Menampilkan daftar percobaan yang tersimpan di sesi aktif dan memilih **Percobaan A** vs **Percobaan B**.
  - [x] Memvalidasi bahwa kedua percobaan berasal dari **mode, latihan, dan target yang sama** (menampilkan peringatan jika tidak setara).
  - [x] Menampilkan pemutar audio kedua percobaan, grafik *overlay* kontur pitch (dalam semitone relatif dengan opsi sumbu waktu asli atau persentase durasi untuk Berbicara; terhadap garis nada target untuk Bernyanyi), dan tabel perbandingan metrik.
  - [x] Menyediakan tombol **Unduh Hasil CSV** dan **Unduh Hasil JSON**.
- **Kriteria Selesai (DoD):** Dua percobaan yang setara dapat dibandingkan secara visual dan numerik, serta dapat diunduh sebagai file CSV/JSON.

#### `TICKET-11` — Dashboard Halaman `4_Metode_dan_Dataset.py` & `5_Evaluasi_Model.py` (`✅ SELESAI`)
- **Prioritas:** `P1 (Wajib Website & Presentasi Tugas)` — **Urutan Eksekusi #9**
- **File yang Dikerjakan:**
  - `pages/4_Metode_dan_Dataset.py`, `pages/5_Evaluasi_Model.py`
- **Daftar Tugas:**
  - [x] Pada `4_Metode_dan_Dataset.py`: menampilkan penjelasan F0, intonasi, nada, cents, tahapan siklus pengenalan pola, serta tabel statistik nyata dari `data/annotations/metadata.csv` (distribusi 328 sampel lintas 4 dataset dan 46 pembicara).
  - [x] Pada `5_Evaluasi_Model.py`: membaca laporan nyata dari `reports/evaluation_report.json` dan `reports/experiment_scenarios.csv`, menampilkan kartu **Accuracy, Balanced Accuracy, Macro-F1**, tabel **Precision/Recall/F1 per kelas**, tabel perbandingan skenario *hyperparameter* SVM & PCA, visualisasi *heatmap* **Confusion Matrix**, serta pembahasan *Overfitting/Underfitting*.
- **Kriteria Selesai (DoD):** Kedua halaman menampilkan data dan metrik evaluasi nyata yang konsisten dengan artefak di `data/` dan `reports/`.

---

### FASE 5 — Pengujian & Dokumentasi Laporan Tugas Pengenalan Pola (`✅ SELESAI`)

#### `TICKET-12` — Pengujian Unit (`pytest`), Validasi Aplikasi & Laporan 5 Bab Tugas Pengenalan Pola (`✅ SELESAI`)
- **Prioritas:** `P2 (Penutup & Laporan Nilai)` — **Urutan Eksekusi #10**
- **File yang Dikerjakan:**
  - `tests/test_*.py`, `README.md`, `LAPORAN_PENGENALAN_POLA.md`
- **Daftar Tugas:**
  - [x] Melengkapi dan menjalankan seluruh pengujian otomatis di `tests/` (`pytest tests/`) hingga 100% lulus (*pass*).
  - [x] Menguji jalannya aplikasi Streamlit (`streamlit run app.py`) pada seluruh 6 halaman tanpa error.
  - [x] Menyusun dokumen laporan akademik lengkap **`LAPORAN_PENGENALAN_POLA.md`** yang mengikuti persis 5 Bab pada PDF tugas kuliah:
    - **Bab 1: Pendahuluan (20%)** — Latar Belakang, Rumusan Masalah, Tujuan Proyek, Batasan Masalah.
    - **Bab 2: Dasar Teori (20%)** — Siklus Pengenalan Pola, Teori $F_0$ & SVM, Teori Metrik Evaluasi.
    - **Bab 3: Metodologi & Penerapan (20%)** — Karakteristik Dataset, Pra-pemrosesan (*missing values*, normalisasi semitone, `StandardScaler`, PCA), Ekstraksi Fitur $F_0$, Pelatihan Model SVM.
    - **Bab 4: Eksperimen, Hasil, & Evaluasi (20%)** — Skenario Eksperimen, Tabel Performa & *Confusion Matrix*, Analisis *Misclassification* & *Overfitting/Underfitting*.
    - **Bab 5: Kesimpulan & Saran (20%)** — Jawaban atas Rumusan Masalah & Rekomendasi Pengembangan.
- **Kriteria Selesai (DoD):** Seluruh unit test lulus, dashboard website berjalan siap pakai, dan laporan 5 Bab siap dikumpulkan/dipresentasikan.
