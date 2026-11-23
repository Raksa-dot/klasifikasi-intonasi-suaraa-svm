# Voice Analyzer — Website Latihan Berbicara dan Bernyanyi

**Voice Analyzer** adalah website latihan berbicara (pola intonasi satu frasa pendek) dan latihan bernyanyi (mengikuti serta menahan nada acuan) berbahasa Indonesia yang dibangun menggunakan **Python & Streamlit**.

Tujuan utama aplikasi ini adalah membantu pengguna memahami perubahan pitch ($F_0$), mengikuti nada acuan, dan membandingkan percobaan latihan melalui umpan balik numerik dan visual yang terukur.

---

## Fitur Utama (MVP)

1. **Latihan Berbicara:**
   - Memilih pola target intonasi (`RISING`, `FALLING`, `LEVEL`, `FALL_RISE`, `RISE_FALL`) dan satu frasa pendek.
   - Visualisasi kontur pitch ($F_0$) terhadap waktu dan kontur relatif dalam semitone.
   - Pengukuran statistik pitch (mean, median, min/max valid, rentang pitch, perubahan awal–akhir, proporsi *voiced*).
   - Klasifikasi pola intonasi menggunakan SVM (diaktifkan hanya setelah model terlatih & tervalidasi tersedia; latihan dasar tetap berfungsi penuh tanpa model SVM).
2. **Latihan Bernyanyi:**
   - Memilih nada target (`A4 = 440 Hz`) dan mendengarkan nada acuan yang disintesis secara lokal.
   - Mengukur ketepatan nada (selisih bertanda dalam **cents**) dan kestabilan nada (variasi pitch selama nada ditahan) secara terpisah.
3. **Bandingkan Percobaan:**
   - Membandingkan dua percobaan dalam sesi yang sama untuk mode, latihan, dan target yang sama.
   - Mengunduh hasil sesi latihan ke format CSV/JSON atas tindakan pengguna.

---

## Struktur Direktori

Lihat rincian arsitektur dan aturan ilmiah di [`AGENTS.md`](./AGENTS.md) serta daftar tugas/milestone di [`TASK.md`](./TASK.md).

```text
voice-analyzer/
├── AGENTS.md
├── TASK.md
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── pages/
├── src/
├── resources/
├── training/
├── models/
├── data/
├── reports/
└── tests/
```

---

## Cara Menjalankan Secara Lokal (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Menjalankan pengujian unit:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```

---

## Privasi dan Batasan Hasil

- Audio diproses pada server tempat aplikasi Python/Streamlit dijalankan dan **tidak** dikirim ke API pihak ketiga.
- Rekaman latihan hanya disimpan sementara di dalam sesi aktif dan tidak disimpan permanen atau dimasukkan ke dataset training secara otomatis.
- Hasil analisis merupakan pengukuran komputasional frekuensi dasar ($F_0$) pada latihan yang dipilih, bukan diagnosis kesehatan/psikologis maupun penilaian kualitas vokal menyeluruh.
