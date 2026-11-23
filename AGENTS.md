# AGENTS.md — Voice Analyzer: Website Latihan Berbicara dan Bernyanyi

> Instruksi untuk AI coding agent/developer yang mengerjakan **Voice Analyzer**. Baca seluruh dokumen ini serta daftar tugas dan milestone pada [`TASK.md`](./TASK.md) sebelum mengubah aplikasi. Periksa repository dan pertahankan implementasi yang masih benar. Instruksi pembangunan di bawah berlaku ketika pengguna meminta implementasi aplikasi; permintaan brainstorming, review, atau revisi dokumen tetap dikerjakan sesuai ruang lingkup permintaan tersebut.

Terakhir diperbarui: **24 November 2026**.

## 1. Ringkasan dan Tujuan Proyek

Bangun **Voice Analyzer**, sebuah **website latihan berbicara dan bernyanyi** berbahasa Indonesia yang dapat digunakan melalui browser laptop dan ponsel. Tujuan utamanya membantu pengguna memahami perubahan pitch, mengikuti nada acuan, dan membandingkan percobaan latihan melalui umpan balik yang terukur.

Alur utama produk:

**Pilih latihan → pahami target/contoh → rekam atau unggah audio → analisis → lihat hasil → coba lagi → bandingkan percobaan.**

Dua jalur latihan:

1. **Latihan Berbicara:** latihan pola intonasi pada satu frasa pendek, visualisasi kontur pitch, perbandingan antarpercobaan, dan klasifikasi pola intonasi menggunakan SVM setelah model yang teruji tersedia.
2. **Latihan Bernyanyi:** mengikuti nada acuan, menahan nada, mengukur selisih terhadap nada target, melihat kestabilan, serta membandingkan percobaan. Latihan interval dan melodi menjadi pengembangan berikutnya.

**MVP latihan harus tetap berguna tanpa model SVM.** Ekstraksi frekuensi dasar, konversi nada, selisih cents, dan visualisasi merupakan pengolahan sinyal/perhitungan. SVM dipakai untuk tugas klasifikasi yang memiliki data berlabel valid, terutama pola intonasi.

Versi awal menganalisis **setelah rekaman selesai**. Tuner langsung saat pengguna bersuara, klasifikasi tipe vokal, dan estimasi kunci musik merupakan fitur lanjutan; jangan menjadikannya syarat selesainya MVP latihan.

Jika proyek juga digunakan untuk penelitian, judul kerja yang dapat dipakai adalah **Klasifikasi Pola Intonasi Ujaran Berdasarkan Fitur F0 Menggunakan SVM pada Website Latihan Suara**. Judul penelitian dan ruang lingkup eksperimen dapat disesuaikan tanpa mengubah fokus latihan pengguna.

## 2. Prinsip Ilmiah dan Batasan Hasil

Gunakan istilah berikut secara konsisten:

- **Pitch/F0:** frekuensi dasar suara, dinyatakan dalam Hz.
- **Intonasi:** perubahan pitch sepanjang satu frasa, misalnya naik, turun, datar, turun-lalu-naik, atau naik-lalu-turun. Bahasa, struktur frasa, dan konteks memengaruhinya.
- **Nada musik:** nada individual yang dipetakan dari frekuensi ke notasi seperti A2, C3, atau D4.
- **Ketepatan nada:** kedekatan pitch yang dinyanyikan terhadap nada target latihan.
- **Kestabilan nada:** variasi pitch selama bagian nada yang ditahan; nada dapat stabil sekaligus meleset dari target.
- **Rentang nada terdeteksi:** nada terendah/tertinggi yang valid dalam rekaman yang dianalisis, bukan otomatis keseluruhan kemampuan vokal pengguna.
- **Rentang vokal dan tipe suara:** memerlukan protokol tes yang sesuai. Kategori Bass, Baritone, Tenor, Contralto/Alto, Mezzo-soprano, dan Soprano tidak cukup ditentukan dari satu kalimat atau hanya min–max F0.
- **Kunci/tangga nada musik:** hubungan kumpulan nada dalam melodi; tidak disimpulkan dari satu nada atau percakapan biasa.

Aturan wajib:

1. Jangan menyebut hasil sebagai diagnosis psikologis, identitas, jenis kelamin, usia, atau kondisi kesehatan.
2. Jangan menyimpulkan kualitas keseluruhan berbicara/bernyanyi hanya dari pitch. Pengukuran MVP belum menilai seluruh aspek artikulasi, ekspresi, teknik vokal, atau musikalitas.
3. Pisahkan **target latihan**, **pengukuran aktual**, **hasil klasifikasi**, dan **saran latihan**. Memilih target `RISING` tidak membuktikan rekaman pengguna benar-benar `RISING`.
4. Jangan membuat data, label, metrik evaluasi, confidence, atau hasil prediksi palsu. Model yang belum ada diberi status yang jujur.
5. Kualitas sinyal, probabilitas frame voiced, skor keputusan SVM, dan ketepatan terhadap nada target adalah besaran berbeda. Jangan menampilkannya sebagai satu persentase keyakinan.
6. Skor keseluruhan, label lulus/gagal, dan batas penilaian harus memiliki definisi, sumber, serta validasi yang jelas. Dahulukan pengukuran yang langsung dapat dijelaskan.
7. Jangan menilai bagian yang tidak memiliki pitch valid. Jika data tidak memadai, tampilkan alasan dan ajakan merekam ulang.
8. Pisahkan data training, validation, dan test berdasarkan pembicara. Rekaman dari pembicara yang sama, termasuk sesi berbeda dan potongan turunannya, tidak boleh bocor antarkelompok evaluasi.
9. Hasil tipe vokal atau key hanya diaktifkan ketika prasyarat data, protokol, dan evaluasinya dipenuhi. Jangan menyimpulkan tipe suara definitif dari satu percobaan.

## 3. Ruang Lingkup Fitur Website

> Rincian daftar tugas implementasi halaman website, milestone pengerjaan, dan checklist penerimaan juga dipisahkan ke dalam [`TASK.md`](./TASK.md).

### 3.1 Beranda dan pemilihan latihan

- Jelaskan manfaat Voice Analyzer dan dua jalur latihan dengan bahasa sederhana.
- Sediakan akses utama ke **Latihan Berbicara** dan **Latihan Bernyanyi**.
- Tampilkan alur latihan singkat dan petunjuk merekam di tempat cukup tenang.
- Jelaskan bahwa hasil adalah pengukuran komputasional terhadap latihan yang dipilih.
- Jangan memenuhi beranda dengan parameter model atau metrik penelitian.

### 3.2 Input audio bersama

- Sediakan rekaman melalui mikrofon browser dan unggahan WAV.
- MP3/FLAC/M4A bersifat opsional jika decoder tersedia dan kegagalan ditangani dengan jelas.
- Validasi isi file yang dapat didekode, ukuran, durasi, sample rate, serta kondisi sinyal.
- Batas ukuran/durasi ditentukan melalui konfigurasi dan ditampilkan sebelum pengguna merekam/mengunggah. Pilih batas awal melalui uji waktu proses dan kebutuhan satu latihan pendek; dokumentasikan alasannya.
- Versi awal memproses satu rekaman setelah pengguna menyelesaikannya, bukan streaming audio.
- Tampilkan pemutar audio, tombol analisis, status proses, rekam ulang, dan reset sesi.
- Jika mikrofon ditolak/tidak tersedia, tampilkan petunjuk izin browser dan pilihan unggah.
- Saat audio, mode, atau target berubah, batalkan hasil lama dan ikat hasil baru pada input yang benar.

### 3.3 Latihan Berbicara

**Unit latihan dan klasifikasi adalah satu frasa pendek.** Jangan memaksakan satu label untuk rekaman panjang berisi banyak frasa dengan pola berbeda. Segmentasi banyak frasa merupakan fitur lanjutan.

Pengalaman latihan:

1. Pengguna memilih pola target dan satu frasa latihan yang disediakan, atau memasukkan teks frasa pendek sendiri.
2. Tampilkan instruksi eksplisit, misalnya “usahakan pitch meningkat pada akhir frasa”, serta ilustrasi kontur. Ilustrasi diberi label sebagai panduan, bukan hasil model.
3. Contoh audio manusia hanya dipakai jika hak penggunaan dan sumbernya jelas. Tidak perlu membuat voice cloning untuk menyediakan contoh latihan.
4. Pengguna merekam/mengunggah satu frasa, mendengarkan ulang, dan menjalankan analisis.
5. Tampilkan kontur pitch serta pengukuran yang valid. Setelah model siap, tambahkan pola intonasi yang diprediksi.
6. Pengguna mencoba kembali dan membandingkan dua percobaan dari latihan yang sama.

Kelas intonasi untuk pengembangan SVM:

- `RISING` — pola pitch meningkat sesuai definisi unit anotasi.
- `FALLING` — pola pitch menurun.
- `LEVEL` — pola pitch relatif datar.
- `FALL_RISE` — pola pitch turun lalu meningkat.
- `RISE_FALL` — pola pitch meningkat lalu menurun.

Pedoman anotasi harus menetapkan bagian frasa yang dinilai, derajat variasi yang diterima, dan perlakuan untuk kontur ambigu. Jangan menyamakan tanda baca dengan ground truth intonasi.

Hasil utama untuk pengguna:

- Target latihan dan grafik kontur pitch terhadap waktu.
- Ringkasan perubahan pitch yang diukur, termasuk perubahan awal–akhir jika datanya memadai.
- Bagian rekaman yang dapat/tidak dapat dianalisis dan alasan kualitasnya.
- Prediksi kelas intonasi **hanya jika** model cocok dan memenuhi prasyarat evaluasi.
- Saran singkat yang terkait pengukuran atau hasil model yang benar-benar tersedia, beserta tombol mencoba lagi.

Detail teknis dapat dilipat: mean/median F0, minimum/maksimum valid, persentil, pitch range, durasi, waveform, dan proporsi voiced. Untuk membandingkan bentuk intonasi, sediakan kontur relatif dalam semitone terhadap median pitch masing-masing rekaman; tetap simpan pitch asli dan informasi waktu.

Jika model belum tersedia, gunakan pesan **“Klasifikasi intonasi belum tersedia; grafik dan pengukuran pitch tetap dapat digunakan untuk latihan.”** Jangan menampilkan kelas dari aturan slope seolah-olah hasil SVM. Pengamatan numerik harus dijelaskan sebagai pengukuran, bukan klasifikasi terlatih.

### 3.4 Latihan Bernyanyi

Prioritas MVP:

1. **Ikuti satu nada:** pengguna memilih nada yang nyaman, mendengarkan acuan, lalu merekam.
2. **Tahan nada:** pengguna mempertahankan nada target selama durasi latihan yang ditentukan dan melihat ketepatan serta kestabilannya.

Nada acuan dapat disintesis secara lokal dari frekuensi yang diketahui. Beri kontrol putar dan berhenti; acuan harus berhenti sebelum perekaman dimulai. Jelaskan penggunaan headphone bila diperlukan agar acuan tidak ikut tertangkap mikrofon. Jangan mengirim audio ke layanan eksternal untuk membuat atau menganalisis nada acuan.

Hasil utama:

- Nama dan frekuensi nada target, dengan konvensi tuning yang jelas.
- Kontur pitch valid dan garis nada target.
- Nada yang terdeteksi dan selisih **bertanda** dalam cents: positif berarti lebih tinggi, negatif berarti lebih rendah.
- Ringkasan selisih terhadap target, ukuran variasi pitch pada bagian stabil, durasi bagian valid, dan proporsi voiced.
- Indikasi waktu bagian yang mendekati/menjauhi target jika metode serta toleransinya dijelaskan.
- Saran latihan berdasarkan pengukuran, misalnya mencoba kembali nada yang sama dan membandingkan hasil.

Tampilkan ketepatan dan kestabilan secara terpisah. Jangan mengabaikan kesalahan satu oktaf dengan hanya membandingkan pitch class. Ukuran variasi tidak boleh langsung disamakan dengan kualitas teknik vokal; vibrato dan transisi harus ditafsirkan sesuai jenis latihan.

Pengembangan berikutnya:

- Latihan interval dan urutan beberapa nada dengan target serta timestamp yang diketahui.
- Analisis melodi, tabel onset/offset, median F0, nama nada, dan selisih cents per segmen.
- Rentang nada terdeteksi, dengan peringatan tentang outlier dan batas rekaman.
- Tes rentang vokal terstruktur, estimasi tipe vokal tervalidasi, dan perkiraan key sebagai fitur lanjutan terpisah.

### 3.5 Bandingkan Percobaan

- Sediakan minimal dua percobaan dalam sesi yang sama, masing-masing dengan player, input, hasil, dan identitas percobaan sendiri.
- Bandingkan percobaan dari **mode, latihan, dan target yang sama**. Jelaskan jika perbandingan tidak setara.
- Berbicara: tampilkan overlay kontur relatif; bila waktu dinormalisasi, tandai sumbu sebagai persentase durasi dan tetap sediakan waktu asli. Normalisasi sederhana tidak membuktikan kesesuaian kata per kata.
- Bernyanyi: bandingkan terhadap nada target yang sama; tampilkan ketepatan, kestabilan, dan cakupan data valid secara terpisah.
- Jangan menyatakan “lebih baik” jika yang dibandingkan hanya perubahan pitch yang tidak memiliki target penilaian.
- Simpan percobaan hanya dalam sesi secara default. Sediakan unduhan hasil CSV/JSON atas tindakan pengguna, tanpa harus menyimpan audio di server secara permanen.
- Riwayat lintas sesi, akun, dan target latihan personal menjadi fitur lanjutan dengan kebijakan penyimpanan yang jelas.

### 3.6 Metode, dataset, dan evaluasi

Halaman **Metode & Dataset** menjelaskan F0, intonasi, nada, cents, cara membaca hasil, pipeline, sumber data, dan keterbatasan dengan bahasa sederhana.

Halaman **Evaluasi Model** menjadi informasi sekunder untuk developer/penelitian. Tampilkan hanya laporan yang benar-benar tersedia: jumlah data/pembicara per split, jumlah per kelas, metode split, macro-F1, balanced accuracy, precision/recall per kelas, accuracy tambahan, confusion matrix, parameter model, versi, dan tanggal training. Untuk pengukuran F0 gunakan metrik referensi yang sesuai bila anotasi tersedia.

Jangan menampilkan angka ilustrasi sebagai hasil nyata. Ketidaktersediaan laporan/model tidak boleh menghalangi latihan dasar.

## 4. Dataset dan Pengelolaan Data

### 4.1 Pisahkan tujuan penggunaan data

1. **Uji pengukuran:** audio dengan referensi F0/nada untuk menguji ekstraksi pitch dan perhitungan numerik.
2. **Training klasifikasi:** rekaman dengan label yang cocok dengan tugas SVM dan sumber anotasi yang jelas.
3. **Uji produk:** rekaman sukarela calon pengguna, termasuk pemula dan perangkat mikrofon sasaran, untuk menilai kegunaan, kegagalan, dan variasi kondisi nyata.

Dataset yang besar tidak otomatis menyediakan label yang dibutuhkan atau membuktikan kecukupan model. Daftar kandidat bukan bukti file telah diunduh, dapat dipasangkan, atau sudah dilatih. Catat status akses dan pemeriksaan yang benar-benar dilakukan.

Pengukuran nada target dan visualisasi dasar dapat dikembangkan tanpa menunggu dataset training klasifikasi. Tetap uji fungsi numeriknya menggunakan sinyal referensi dan materi suara yang sesuai.

### 4.2 Prioritas sumber awal

| Sumber | Kegunaan | Batas penggunaan dalam proyek |
| --- | --- | --- |
| [PTDB-TUG](https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html) | Referensi pitch ujaran dan pengujian F0 | Berisi penutur bahasa Inggris; bukan sumber siap pakai untuk lima label intonasi bahasa Indonesia. |
| [Common Voice Scripted Speech](https://github.com/common-voice/cv-dataset/blob/main/datasets/scripted-speech/README.md) — subset Indonesia | Bahan audio ujaran dan transkrip | Skema umumnya tidak menyediakan lima label intonasi proyek. Periksa rilis/akses resmi terbaru dan lakukan anotasi yang sesuai. |
| [VocalSet v1.2](https://zenodo.org/records/1442513) + [Annotated-VocalSet](https://zenodo.org/records/7061507) | Audio bernyanyi dan anotasi F0/nada/waktu | Pasangkan audio dan anotasi menurut identitas serta versi. Penyanyi profesional belum mewakili seluruh pengguna pemula. |
| [GTSinger](https://huggingface.co/datasets/GTSinger/GTSinger) | Eksperimen tambahan nyanyian dan anotasi terkait | Periksa metadata, bahasa, versi, serta tugas yang cocok. Label teknik bernyanyi bukan lima kelas intonasi proyek. |

VocalSet memang menyediakan pemetaan tipe suara penyanyi dalam dokumentasinya. Jangan menyatakan label itu tidak ada; periksa cakupan kelas, jumlah penyanyi, protokol, dan generalisasi sebelum menggunakannya untuk klasifikasi pengguna baru.

### 4.3 Kandidat tambahan, bukan unduhan wajib MVP

- [Saarbrücken Voice Database](https://stimmdb.coli.uni-saarland.de/) — hanya gunakan subset yang sesuai tujuan pengukuran dan ketentuan aksesnya; bukan dasar untuk klaim kesehatan pada website.
- [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) — eksperimen ujaran tambahan; akses/lisensi harus diperiksa.
- [MIR-1K](https://zenodo.org/records/3532216) dan [iKala](https://zenodo.org/records/3532214) — kandidat evaluasi pitch/melodi; periksa audio, anotasi, dan ketentuan rilis.
- [NUS-48E](https://huggingface.co/datasets/guey-khala-mari/NUS-48E) — kandidat pembanding ujaran/nyanyian; verifikasi asal, versi, serta hak penggunaan sebelum dipakai.

Periksa lisensi, atribusi, ketentuan akses, dan batas distribusi setiap sumber. Jangan menggabungkan dataset dengan asumsi label/protokolnya sama atau mengunduh semua kandidat tanpa kebutuhan milestone.

### 4.4 Prosedur anotasi intonasi

- Tetapkan satu frasa sebagai unit, bagian kontur yang dinilai, definisi lima kelas, dan contoh batas antarkelas.
- Gunakan lebih dari satu penilai pada subset validasi anotasi, ukur kesepakatan, dan selesaikan perbedaan dengan prosedur tercatat.
- Sediakan status ambigu/tidak layak; data demikian tidak dipaksa masuk salah satu kelas training.
- Catat target latihan terpisah dari label aktual. Target yang diminta, tanda baca, atau aturan slope tidak boleh otomatis menjadi ground truth.
- Label yang berasal dari algoritma perlu dinyatakan sebagai label otomatis dan divalidasi; jangan mengklaimnya sebagai anotasi manusia.
- Catat asal pembicara dan hubungan semua potongan, sesi, dan augmentasi. Seluruh turunan satu pembicara mengikuti split pembicaranya.
- Kecukupan akhir dilihat dari kualitas label, cakupan pembicara/kondisi, kinerja pada pembicara baru, dan analisis kesalahan; jangan menjanjikan akurasi berdasarkan jumlah jam saja.

### 4.5 Direktori dan metadata

```text
data/
├── raw/                 # Data asli; tidak diedit
│   ├── speech/
│   └── singing/
├── processed/
├── annotations/
├── features/
└── README.md            # Sumber, versi, lisensi, akses, protokol, dan split
```

Contoh kolom metadata; gunakan hanya yang relevan dan dokumentasikan skemanya:

```csv
file_path,dataset,dataset_version,speaker_id,session_id,source_clip_id,song_id,split,task,language,exercise_id,target_label,intonation_label,annotation_status,annotator_id,annotation_source,vocal_type_label,music_key_label
```

Label yang tidak tersedia harus kosong/`NA`, bukan ditebak. `speaker_id` harus konsisten lintas sesi; jangan membuat kelompok baru untuk sesi berbeda dari orang yang sama. Jika identitas penyanyi dan lagu sama-sama harus terpisah, rancang split yang mencegah overlap keduanya; gabungan pasangan penyanyi–lagu saja tidak menjamin hal ini.

Jangan commit audio dataset mentah, unggahan pengguna, atau model/data yang tidak boleh didistribusikan. Dokumentasikan proses memperoleh data, alih-alih membundelnya tanpa izin.

### 4.6 Data pengguna untuk evaluasi atau training

Rekaman latihan tidak otomatis masuk dataset. Pengumpulan untuk penelitian/training memerlukan pilihan ikut serta yang jelas dan terpisah dari penggunaan latihan. Dokumentasikan tujuan, data yang disimpan, masa retensi, dan cara penghapusan. MVP tidak membutuhkan pengumpulan tersebut untuk dapat berjalan.

## 5. Pipeline Pengolahan Audio

### 5.1 Validasi dan pemeriksaan kualitas

- Pastikan file benar-benar dapat didekode; ekstensi saja tidak cukup.
- Periksa batas ukuran/durasi sebelum komputasi mahal dan tangani audio kosong, rusak, terlalu pendek, hening, clipping berat, serta tidak memiliki cukup frame voiced.
- Ukur clipping dan energi dari sinyal asli sebelum normalisasi yang dapat menutupi masalahnya.
- Ambang kualitas harus terpusat, terdokumentasi, dan diuji. Pisahkan kualitas sinyal dari prediksi kelas atau ketepatan nada.
- Tampilkan alasan konkret yang didukung pemeriksaan. Jangan mengklaim deteksi kebisingan/iringan dominan jika algoritmanya belum diimplementasikan dan diuji.
- Status “memadai” berarti cukup untuk pengukuran tertentu, bukan jaminan kualitas rekaman atau seluruh hasil model.

### 5.2 Preprocessing

1. Baca audio dan sample rate asli; pertahankan informasi durasi/waktu.
2. Konversi ke mono untuk pipeline awal dan resample ke sample rate internal konsisten, misalnya 22.050 Hz.
3. Pertahankan sinyal asli untuk pemeriksaan energi/clipping; hilangkan DC offset bila diperlukan.
4. Jika melakukan trim, simpan offset agar grafik, player, dan feedback memakai timestamp yang benar.
5. Jangan menerapkan noise reduction agresif secara default karena dapat mengubah kontur pitch.
6. Gunakan konfigurasi frame/hop yang terdokumentasi dan sesuai rentang pitch; uji kemampuan mendeteksi nada rendah dan tinggi.

### 5.3 Estimasi pitch/F0

- Gunakan `librosa.pyin` sebagai implementasi awal; Praat/Parselmouth dapat menjadi pembanding.
- Sediakan `fmin`, `fmax`, `frame_length`, dan `hop_length` berdasarkan mode. Jangan memakai rentang sempit untuk semua suara.
- Simpan F0 per frame, timestamp, voiced mask/probability, dan konfigurasi ekstraksi.
- Keluarkan frame non-voiced dan nilai tidak valid dari statistik/perbandingan.
- Jangan mengisi jeda panjang dengan pitch rekaan. Interpolasi kontur untuk fitur hanya pada bagian yang layak dengan aturan gap yang jelas; grafik harus mempertahankan informasi bagian tidak valid.
- Saring outlier dengan hati-hati, dokumentasikan aturannya, dan jangan menghapus transisi/vibrato yang memang ada.
- Jika data valid terlalu sedikit, batalkan ringkasan/klasifikasi yang tidak dapat dipertanggungjawabkan.

### 5.4 Fitur, perbandingan, dan ringkasan

Fitur intonasi awal: mean/median, standard deviation, min/max valid, pitch range, persentil, slope, selisih awal–akhir, ringkasan bentuk kontur, serta cakupan voiced. Selisih awal–akhir sebaiknya memakai ringkasan bagian awal/akhir yang valid dan terdokumentasi, bukan satu frame ekstrem.

Bandingkan fitur Hz dengan fitur kontur relatif dalam semitone. Fitur tambahan seperti MFCC/RMS hanya ditambahkan bila membantu eksperimen yang terukur; jitter/shimmer/HNR tidak diperlukan untuk MVP latihan.

Untuk bernyanyi, hitung selisih terhadap target dan variasi pada segmen nada yang relevan. Jangan menghitung stabilitas satu nada pada rekaman melodi campuran. Simpan cakupan data valid agar rekaman yang sebagian besar gagal terdeteksi tidak tampak berhasil hanya karena beberapa frame yang tersisa.

Satukan kode ekstraksi untuk website, training, dan evaluasi agar parameter/urutan fiturnya konsisten. Identitas hasil minimal mencakup input, mode, target/latihan, versi extractor, dan konfigurasi.

## 6. Nada Acuan, Konversi Nada, dan Umpan Balik

### 6.1 Frekuensi dan notasi musik

Gunakan tuning default A4 = 440 Hz yang terdokumentasi:

```text
midi = 69 + 12 * log2(frequency_hz / 440.0)
```

Bulatkan ke semitone terdekat hanya untuk nama nada. Pertahankan frekuensi/nilai MIDI kontinu untuk pengukuran selisih. Abaikan F0 tidak valid dan nilai `<= 0`. Nada acuan harus dihasilkan dari frekuensi yang sama dengan target analisis.

### 6.2 Selisih terhadap nada target

```text
cents_error = 1200 * log2(f0_hz / target_hz)
```

- Nilai positif: lebih tinggi dari target; negatif: lebih rendah; nol: tepat pada target.
- Gunakan nada target latihan, bukan selalu nada terdekat. Bedakan selisih terhadap target dari deviasi terhadap semitone terdekat.
- Jangan membuang perbedaan oktaf: satu oktaf adalah selisih 1200 cents.
- Laporkan ringkasan yang jelas, misalnya median selisih bertanda untuk arah penyimpangan dan median selisih absolut untuk kedekatan terhadap target.
- Untuk kestabilan nada yang ditahan, gunakan ukuran variasi yang terdokumentasi, misalnya median absolute deviation dalam cents terhadap median pitch segmen. Ukuran ini terpisah dari selisih terhadap target.
- Jika menampilkan proporsi frame dalam toleransi, jelaskan toleransi, denominator frame valid, dan cakupan voiced. Itu hasil pengukuran dengan aturan tertentu, bukan confidence model atau penilaian keseluruhan kemampuan.
- Jangan menetapkan batas universal “bagus/buruk” tanpa dasar dan validasi. Nilai konfigurasi awal harus disebut sebagai parameter latihan yang dapat dievaluasi.

### 6.3 Saran latihan

- Hubungkan saran dengan pengukuran yang tersedia: arah selisih nada, variasi selama tahan nada, atau perubahan pitch pada bagian frasa.
- Gunakan template feedback yang dapat ditelusuri ke pengukuran; MVP tidak membutuhkan API generatif.
- Jika model belum ada, saran berbicara dapat mengajak pengguna melihat/membandingkan kontur, tanpa menyatakan kelas intonasi telah diprediksi.
- Kegagalan deteksi menghasilkan petunjuk merekam ulang, bukan angka nol yang tampak sebagai keberhasilan.

### 6.4 Fitur musik lanjutan

Gabungkan frame menjadi segmen nada dengan smoothing yang tidak memecah vibrato menjadi banyak notasi palsu. Evaluasi onset/offset dan nama nada terhadap referensi.

Estimasi key dapat memakai histogram pitch class/chroma dan profil major/minor terdokumentasi sebagai baseline. Aktifkan hanya untuk melodi dengan bukti nada yang memadai; batasi klaim sebagai perkiraan dan tangani ambiguitas. Jangan menjalankannya pada speech atau satu nada. SVM key memerlukan data melodi berlabel serta evaluasi terpisah.

## 7. Model Machine Learning SVM

### 7.1 Kesiapan data dan model

SVM intonasi adalah pengembangan setelah fondasi latihan. Sebelum training, pastikan unit frasa, label aktual, identitas speaker, jumlah data/kelas, dan prosedur anotasi diperiksa. Jangan menganggap dataset referensi F0 otomatis cukup untuk klasifikasi intonasi.

Pipeline awal:

```text
StandardScaler → SVC(kernel="rbf", class_weight="balanced")
```

- Gunakan `Pipeline`; scaler/seleksi fitur di-fit hanya pada bagian training setiap fold.
- Tuning `C`/`gamma` dengan `GridSearchCV` dan pembagian berkelompok berdasarkan speaker, misalnya `GroupKFold` atau `StratifiedGroupKFold` jika distribusi memungkinkan.
- Test akhir dipisahkan dan tidak dipakai untuk memilih model/parameter/ambang.
- Bandingkan dengan baseline sederhana yang diberi identitas jelas. Aturan yang sama untuk membuat label dan memprediksi label tidak membuktikan generalisasi intonasi.
- Jangan mengaktifkan `probability=True` lalu menyajikan persentase sebagai confidence tanpa pemeriksaan kalibrasi dan desain evaluasi yang sesuai.

### 7.2 Artefak dan metadata

- `intonation_svm.joblib` — model intonasi setelah validasi.
- `vocal_range_svm.joblib` — opsional, untuk tugas/protokol rentang atau tipe vokal yang didefinisikan dan tervalidasi.
- `music_key_svm.joblib` — opsional, setelah tersedia data key yang sesuai.

Setiap model menyimpan versi, definisi label/tugas, daftar fitur beserta urutan, versi extractor, parameter audio, versi Python/library, sumber/versi/lisensi data, metode split, tanggal training, dan laporan evaluasi terkait. Tolak model yang metadata fiturnya tidak cocok. Muat hanya artefak model tepercaya milik aplikasi; jangan menerima unggahan model pengguna untuk dieksekusi.

### 7.3 Evaluasi dan aktivasi di website

- Laporkan jumlah sampel dan pembicara per kelas/split, macro-F1, balanced accuracy, precision/recall per kelas, confusion matrix, serta accuracy sebagai tambahan.
- Uji pada pembicara baru dan periksa kegagalan pada perangkat/kondisi pengguna sasaran.
- Jika evaluasi belum memadai atau model tidak cocok, tampilkan status dan tetap sediakan latihan pengukuran dasar.
- Kriteria aktivasi model ditetapkan berdasarkan tujuan penggunaan dan evaluasi nyata; jangan membuat angka keberhasilan agar UI tampak siap.
- Untuk model bernyanyi, cegah kebocoran penyanyi/lagu sesuai desain evaluasi dan catat overlap yang memang masih ada.
- Training melalui skrip terpisah. Website hanya melakukan ekstraksi dan inferensi; tidak melatih ulang saat dibuka atau saat audio diunggah.

## 8. Teknologi dan Arsitektur Website

### 8.1 Stack MVP

- Python 3.11 atau versi yang kompatibel dengan dependency yang diperiksa.
- Streamlit untuk antarmuka website, input mikrofon, upload, player, dan state sesi.
- Librosa, NumPy, SciPy, dan SoundFile untuk pengolahan audio/perhitungan.
- Pandas untuk tabel/ekspor; Matplotlib atau Plotly untuk grafik.
- scikit-learn dan Joblib untuk model offline dan inferensi setelah tersedia.
- Parselmouth opsional sebagai pembanding.

Mulai sebagai satu website Streamlit dengan modul Python terpisah. Jangan menambahkan Laravel, REST API, database, login, atau layanan eksternal tanpa kebutuhan yang disepakati. Pilih/pin versi dependency yang kompatibel dan verifikasi API aktual, termasuk komponen mikrofon.

### 8.2 Lokasi pemrosesan dan hosting

Browser mengakses frontend; Python memproses audio pada mesin/server tempat aplikasi dijalankan. Saat dijalankan lokal, server ada di komputer developer/pengguna. Saat di-host, audio dikirim dari browser ke server website tersebut. Jangan menyebut versi yang di-host sebagai pemrosesan sepenuhnya di perangkat pengguna.

Untuk akses publik melalui mikrofon, siapkan **HTTPS**, izin mikrofon browser, dan fallback upload. `localhost` dapat dipakai untuk pengembangan lokal. Akses dari browser HP harus diuji pada alamat yang sesuai dan aman, bukan diasumsikan berhasil karena browser laptop bekerja.

Dokumentasikan cara menjalankan lokal dan menyiapkan hosting. Publikasi/deployment dilakukan ketika diminta, dengan lingkungan yang mendukung dependency audio, batas upload, dan resource komputasi. Website tidak perlu deployment publik untuk menguji perhitungan secara lokal, tetapi kesiapan publik memiliki kriteria tersendiri pada bagian 15.

### 8.3 Diagram arsitektur

```text
Browser laptop/HP
  └── Streamlit: pilih latihan, acuan, rekam/upload, player
        └── Server Python
              ├── Validasi → preprocessing → F0 + kualitas
              ├── Speech: kontur relatif + pengukuran frasa
              │     └── SVM intonasi, jika siap
              ├── Singing: target → cents + kestabilan
              └── Hasil sesi → perbandingan → unduhan hasil

Workflow offline yang terpisah
  Data + anotasi → ekstraksi → split speaker → tuning/evaluasi
        └── Model + metadata + laporan → inferensi website
```

## 9. Struktur Folder yang Ditargetkan

Sesuaikan dengan repository yang ada; jangan menumpuk semuanya dalam `app.py` atau membuat semua modul fitur lanjutan sebelum diperlukan.

```text
voice-analyzer/
├── AGENTS.md
├── TASK.md
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── pages/
│   ├── 1_Latihan_Berbicara.py
│   ├── 2_Latihan_Bernyanyi.py
│   ├── 3_Bandingkan_Percobaan.py
│   ├── 4_Metode_dan_Dataset.py
│   └── 5_Evaluasi_Model.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── audio_io.py
│   ├── validation.py
│   ├── preprocessing.py
│   ├── pitch.py
│   ├── features.py
│   ├── notes.py
│   ├── reference_tones.py
│   ├── exercises.py
│   ├── feedback.py
│   ├── comparison.py
│   ├── prediction.py
│   ├── visualization.py
│   └── export.py
├── resources/
│   └── exercises.json
├── training/
│   ├── extract_dataset_features.py
│   ├── split_dataset.py
│   ├── train_intonation.py
│   └── evaluate.py
├── models/
│   └── README.md
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   ├── annotations/
│   └── features/
├── reports/
│   └── .gitkeep
└── tests/
    ├── test_audio_validation.py
    ├── test_pitch.py
    ├── test_notes.py
    ├── test_feedback.py
    ├── test_comparison.py
    └── test_prediction.py
```

Definisi latihan minimal memuat ID/versi, mode, instruksi, teks frasa atau nada target, durasi target jika relevan, dan sumber/jenis acuan. Nilai target dipisahkan dari hasil analisis aktual. Tambahkan modul training tipe vokal/key hanya saat tugas lanjutan itu benar-benar dikerjakan.

## 10. Ketentuan UI/UX

- Bahasa Indonesia, tampilan rapi, responsif, warna netral dengan satu aksen, dan navigasi yang jelas.
- Navigasi utama: Beranda, Latihan Berbicara, Latihan Bernyanyi, Bandingkan Percobaan; metode/dataset dan evaluasi menjadi informasi sekunder.
- Utamakan target latihan, petunjuk, tombol rekam, hasil singkat, dan ajakan mencoba lagi.
- Tempatkan parameter/angka teknis dalam detail yang dapat dibuka; tampilkan istilah Hz/cents dengan penjelasan singkat ketika diperlukan.
- Grafik memiliki label sumbu, satuan, timestamp atau persentase waktu yang jelas, serta legenda target/percobaan.
- Gunakan label/teks selain warna untuk membedakan hasil. Kontrol dan pesan harus tetap terbaca pada layar HP.
- Tampilkan progress saat komputasi dan cegah pengiriman analisis berulang yang tidak perlu.
- Jangan menjanjikan analisis instan sebelum mengukur waktu proses. Batas durasi dan informasi proses harus sesuai hasil pengujian.
- Jangan menampilkan hasil lama sebagai hasil input/target baru atau kartu hasil valid saat analisis gagal.
- Gunakan error bahasa Indonesia yang membantu, tanpa traceback Python kepada pengguna.
- Status model belum tersedia ditempatkan pada fitur klasifikasi; pengguna tetap dapat memakai fitur latihan yang berfungsi.
- Hindari istilah “suara buruk” atau peringkat kemampuan tanpa metode penilaian yang tervalidasi.

## 11. Privasi dan Keamanan

- Jelaskan pemrosesan audio di server aplikasi sesuai bagian 8; jangan mengirim audio ke API analisis eksternal/pihak ketiga.
- Jangan menyimpan rekaman latihan secara permanen secara default atau otomatis menggunakannya untuk training.
- Isolasi audio dan hasil antar sesi. Hindari global/shared cache untuk audio pengguna yang dapat memperpanjang retensi atau membuka akses antarpengguna.
- Jika temporary file diperlukan, gunakan nama/path aman, batasi akses dan masa hidupnya, serta bersihkan pada alur normal dan error.
- Reset harus membersihkan input audio, hasil turunan, kedua percobaan, dan file sementara yang dikelola aplikasi; jangan sekadar menyembunyikan kartu hasil.
- Jangan mencatat audio mentah atau isi sensitif pengguna dalam log.
- Validasi upload; jangan mengeksekusi file unggahan atau memuat model Joblib dari pengguna.
- Unduhan hasil adalah tindakan pengguna. Penyimpanan riwayat server, pengumpulan dataset, dan retensi audio memerlukan persetujuan yang terpisah serta mekanisme penghapusan.
- Uji isolasi sesi dan siklus reset; dokumentasikan batas pembersihan yang benar-benar dapat dijamin oleh implementasi/framework.

## 12. Kualitas Implementasi dan Pengujian

1. Inspeksi repository, Python, dependency, dan implementasi yang ada sebelum mengubah kode. Jika workspace adalah proyek lain, jangan menimpa kode/instruksinya sebagai asumsi.
2. Pisahkan UI, pemrosesan audio, definisi latihan, feedback, perbandingan, training, dan inferensi.
3. Gunakan konfigurasi terpusat untuk sample rate, frame/hop, pitch bounds, batas input, toleransi latihan, dan lokasi artefak. Hindari path absolut komputer developer.
4. Gunakan type hints pada fungsi utama, docstring untuk fungsi kompleks, dan identitas versi extractor/model/latihan.
5. Tangani model hilang, metadata tidak cocok, audio tidak valid, data voiced kurang, serta perubahan input/target yang membatalkan hasil lama.
6. Tetapkan random seed eksperimen dan jangan mengubah data mentah.
7. Tambahkan pengujian bermakna untuk rumus dan alur yang berisiko salah; tidak perlu test yang sekadar menyalin implementasi.
8. Uji frekuensi/nada referensi, arah selisih cents, selisih satu oktaf, kestabilan vs penyimpangan target, timestamp, cakupan voiced, serta perbandingan latihan yang tidak setara.
9. Sinyal sintetis membuktikan perhitungan dasar; tetap evaluasi F0 pada audio suara nyata beranotasi. Catat toleransi, metode referensi, dan hasilnya.
10. Uji hening, noise, terlalu pendek, clipping, file rusak, format salah, dan input valid. Kegagalan tidak boleh menghasilkan angka atau label yang tampak sukses.
11. Uji browser HP/laptop: izin/penolakan mikrofon, upload, player, grafik, perubahan target, dua percobaan, reset, dan unduhan.
12. Uji dengan calon pengguna pemula/perangkat sasaran atas pilihan ikut serta mereka. Periksa apakah feedback dipahami dan membantu menentukan percobaan berikutnya; jangan mengklaim efektivitas latihan hanya dari akurasi model.
13. Jalankan pemeriksaan yang sesuai perubahan dan perbarui README. Laporkan apa yang benar-benar diuji, termasuk browser/perangkat dan keterbatasan.

## 13. Urutan Pengerjaan (Milestone) & Kriteria Penerimaan

Lihat rincian lengkap **Milestone 1–8**, daftar keperluan implementasi halaman website, serta **Checklist Kriteria Penerimaan (15.1, 15.2, 15.3)** pada file [`TASK.md`](./TASK.md).

## 14. Perintah Pengembangan yang Ditargetkan

Contoh Windows PowerShell, disesuaikan dengan implementasi aktual:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Pemanggilan interpreter virtual environment langsung tidak memerlukan perubahan execution policy PowerShell. Dokumentasikan versi Python/dependency yang benar-benar dipakai dan seluruh argumen yang diperlukan.

Training offline setelah modul/data tersedia:

```powershell
.\.venv\Scripts\python.exe -m training.extract_dataset_features
.\.venv\Scripts\python.exe -m training.train_intonation
.\.venv\Scripts\python.exe -m training.evaluate
```

Dokumentasikan pula perintah test yang sesuai runner aktual. Jangan menjalankan unduhan dataset besar atau training sebagai efek samping membuka website.

## 15. Instruksi Awal untuk AI Coding Agent

1. Pahami permintaan terbaru pengguna: apakah implementasi, perbaikan, review, atau revisi dokumen. Jangan menjalankan seluruh roadmap ketika pengguna hanya meminta pembahasan/dokumentasi.
2. Saat implementasi diminta, inspeksi target repository, stack, versi, dan fitur yang sudah ada. Tentukan target Voice Analyzer dengan benar dan pertahankan pekerjaan yang masih berfungsi.
3. Buat rencana singkat sesuai kondisi aktual, lalu kerjakan milestone kecil yang dapat diuji berdasarkan [`TASK.md`](./TASK.md). Mulai dari website dan input audio bila belum ada.
4. Lanjutkan ke pengukuran teruji, latihan terpandu, dan perbandingan percobaan. Jangan menunggu SVM untuk menyelesaikan fitur latihan yang tidak memerlukannya.
5. Jalankan jalur anotasi/training SVM secara terpisah setelah prasyaratnya diperiksa; jangan mengarang hasil untuk melengkapi UI.
6. Pada setiap kemajuan, verifikasi fungsi yang berubah dan perbarui README serta status di [`TASK.md`](./TASK.md) seperlunya. Laporkan fitur yang benar-benar berjalan, yang menunggu data/model, serta yang belum diuji atau dipublikasikan.
7. Prioritaskan umpan balik yang dipahami pengguna dan dapat ditindaklanjuti. Hasil yang valid dan status “data belum memadai” lebih berguna daripada penilaian yang tampak meyakinkan tanpa dasar.
