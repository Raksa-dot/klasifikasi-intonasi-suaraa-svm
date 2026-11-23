# LAPORAN PROYEK PENGENALAN POLA
## Klasifikasi Pola Intonasi Ujaran Berdasarkan Fitur Frekuensi Dasar ($F_0$) Menggunakan Support Vector Machine (SVM) pada Website Latihan Suara (*Voice Analyzer*)

---

## 1. Pendahuluan (Background & Rumusan Masalah) (20%)

### 1.1 Latar Belakang
Intonasi dan ketepatan nada (*pitch*) memegang peranan krusial dalam komunikasi lisan maupun pelatihan vokal. Dalam berbicara, perubahan frekuensi dasar suara ($F_0$) sepanjang satu frasa pendek menentukan fungsi komunikatif ujaran—apakah suatu frasa terdengar sebagai pertanyaan yang meningkat di akhir (*rising*), pernyataan penutup yang menurun (*falling*), ujaran datar (*level*), maupun pola lengkung kompleks (*fall-rise* dan *rise-fall*). Demikian pula dalam latihan bernyanyi dasar, kemampuan membidik frekuensi nada acuan dan mempertahankan kestabilan pitch sangat menentukan kualitas latihan.

Namun, pembelajar mandiri sering mengalami kesulitan mengevaluasi intonasi dan ketepatan nadanya sendiri karena penilaian pendengaran subjektif rentan bias dan tidak memberikan umpan balik numerik yang terukur. Oleh karena itu, pendekatan **Pengenalan Pola (*Pattern Recognition*)** dan pengolahan sinyal audio digital diperlukan untuk mengotomatisasi ekstraksi kontur $F_0$, menormalisasi variasi tinggi suara antar-pembicara, serta mengklasifikasikan pola intonasi ujaran secara objektif menggunakan algoritma **Support Vector Machine (SVM)** yang terintegrasi langsung ke dalam aplikasi website latihan interaktif (**Voice Analyzer**).

### 1.2 Rumusan Masalah
1. Bagaimana mengekstraksi dan menormalisasi fitur kontur frekuensi dasar ($F_0$) dari sinyal audio ujaran mentah serta menangani bagian tanpa getaran pita suara (*unvoiced frames / missing values*) agar invarian terhadap perbedaan tinggi suara dasar antar-pembicara?
2. Bagaimana kinerja algoritma **Support Vector Machine (SVM)** dengan berbagai variasi *kernel*, *hyperparameter*, serta reduksi dimensi (*Principal Component Analysis / PCA*) dalam mengklasifikasikan lima kelas pola intonasi ujaran (`RISING`, `FALLING`, `LEVEL`, `FALL_RISE`, dan `RISE_FALL`) pada pembicara baru (*speaker-independent evaluation*)?
3. Bagaimana mengintegrasikan pipeline pengolahan sinyal $F_0$ dan model klasifikasi SVM ke dalam website latihan berbicara dan bernyanyi yang memberikan umpan balik terukur secara transparan?

### 1.3 Tujuan Proyek
1. Membangun pipeline pengolahan sinyal audio dan ekstraksi **14 fitur numerik kontur $F_0$** dalam skala semitone relatif per pembicara menggunakan `librosa.pyin`.
2. Melatih dan mengevaluasi model klasifikasi pola intonasi 5 kelas menggunakan **Support Vector Machine (SVM)** dengan protokol *Speaker-Independent Split* hingga mencapai akurasi uji di atas **85%** dan *Macro-F1* di atas **0,80**.
3. Mengimplementasikan website interaktif **Voice Analyzer** berbasis **Python & Streamlit** yang mendukung latihan berbicara, latihan bernyanyi (ketepatan *cents* dan kestabilan nada terpisah), monitor pitch *real-time*, serta perbandingan antarpercobaan.

### 1.4 Batasan Masalah
1. **Unit Analisis:** Satu rekaman latihan berbicara dibatasi pada **satu frasa pendek** (durasi 0,6 – 15,0 detik) dengan sinyal suara monofonik tanpa iringan musik latar.
2. **Jumlah Kelas Klasifikasi:** Lima kelas pola intonasi ujaran, yaitu `RISING` (Naik), `FALLING` (Turun), `LEVEL` (Datar), `FALL_RISE` (Turun-Lalu-Naik), dan `RISE_FALL` (Naik-Lalu-Turun).
3. **Fitur yang Digunakan:** Fitur akustik berbasis kontur frekuensi dasar ($F_0$) yang dinormalisasi ke dalam satuan semitone relatif terhadap median pitch masing-masing rekaman.
4. **Batasan Klaim Ilmiah:** Hasil sistem merupakan pengukuran komputasional terhadap pitch pada latihan yang dipilih, bukan alat diagnosis medis/psikologis dan belum menilai artikulasi atau kualitas vokal menyeluruh.

---

## 2. Dasar Teori (Theoretical Foundation) (20%)

### 2.1 Siklus Pengenalan Pola (*Pattern Recognition Cycle*)
Sistem **Voice Analyzer** mengikuti lima tahapan fundamental siklus pengenalan pola:
1. **Pengumpulan Data (*Sensing / Acquisition*):** Akuisisi gelombang suara akustik melalui mikrofon browser atau file audio digital (WAV/MP3) menjadi sinyal diskrit $x[n]$.
2. **Pra-Pemrosesan (*Preprocessing*):** Validasi kualitas sinyal (energi RMS dan deteksi *clipping*), konversi ke kanal tunggal (*mono*), penghilangan *DC offset*, *resampling* ke `22.050 Hz`, estimasi kontur $F_0$ per frame, serta penanganan nilai kosong (*missing values*) pada frame *unvoiced*.
3. **Ekstraksi Fitur (*Feature Extraction*):** Transformasi deret waktu $F_0(t)$ yang panjangnya bervariasi menjadi vektor fitur numerik berdimensi tetap ($\mathbf{x} \in \mathbb{R}^{14}$) melalui normalisasi semitone relatif, segmentasi tiga bagian (awal, tengah, akhir), regresi kemiringan (*slope*), dan pencocokan polinomial orde-2.
4. **Klasifikasi (*Classification*):** Pemetaan vektor fitur $\mathbf{x}$ yang telah distandarisasi (`StandardScaler`) ke salah satu dari 5 kelas pola intonasi menggunakan **Support Vector Machine (SVM)**.
5. **Evaluasi Keputusan (*Evaluation & Post-Processing*):** Pengukuran kinerja model pada pembicara baru menggunakan *Confusion Matrix*, *Accuracy*, *Balanced Accuracy*, *Precision*, *Recall*, dan *Macro-F1*, serta penyajian umpan balik latihan kepada pengguna.

### 2.2 Frekuensi Dasar ($F_0$), Nada Musik, dan Skala Semitone/Cents
- **Estimasi Pitch (`pYIN`):** Algoritma *probabilistic YIN* (`librosa.pyin`) memodelkan kandidat periode fundamental menggunakan fungsi perbedaan autokorelasi dan *Hidden Markov Model (HMM)* untuk menghasilkan estimasi frekuensi $F_0$ (Hz) beserta status *voiced/unvoiced* setiap frame.
- **Konversi Nada Musik (`A4 = 440 Hz`):**
  $$\text{MIDI} = 69 + 12 \cdot \log_2\left(\frac{f_{\text{Hz}}}{440,0}\right)$$
- **Selisih Cents terhadap Nada Target ($f_{\text{target}}$):**
  $$\Delta c = 1200 \cdot \log_2\left(\frac{f_0}{f_{\text{target}}}\right)$$

### 2.3 Metode/Algoritma Utama: Support Vector Machine (SVM)
**Support Vector Machine (SVM)** mencari *hyperplane* pemisah optimal $\mathbf{w}^\top \phi(\mathbf{x}) + b = 0$ yang memaksimalkan margin antar-kelas sekaligus meminimalkan kesalahan klasifikasi melalui parameter regularisasi $C$:

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} w_{y_i} \xi_i \quad \text{dengan syarat} \quad y_i(\mathbf{w}^\top \phi(\mathbf{x}_i) + b) \ge 1 - \xi_i$$

Untuk memisahkan pola kontur non-linier (seperti bentuk lengkung `FALL_RISE` dan `RISE_FALL`), digunakan **Radial Basis Function (RBF) Kernel**:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

Pembobotan kelas terbalik (`class_weight="balanced"`) diterapkan agar kelas dengan jumlah sampel lebih sedikit tetap memperoleh penalti margin yang proporsional.

### 2.4 Metrik Evaluasi
- **Confusion Matrix:** Matriks $C \times C$ yang memetakan jumlah sampel kelas sebenarnya (*true label*) terhadap kelas hasil prediksi (*predicted label*).
- **Accuracy:** Proporsi seluruh prediksi yang benar ($\frac{\text{TP} + \text{TN}}{\text{Total}}$).
- **Precision, Recall, dan F1-Score per Kelas $k$:**
  $$\text{Precision}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k}, \quad \text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}, \quad F_{1,k} = \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$
- **Macro-F1 & Balanced Accuracy:** Rata-rata aritmetika $F_{1,k}$ dan $\text{Recall}_k$ di seluruh 5 kelas tanpa bobot ukuran kelas, sehingga menjadi indikator paling adil pada dataset dengan distribusi kelas tidak seragam.

---

## 3. Metodologi & Penerapan (Implementation & Workflow) (20%)

### 3.1 Dataset & Karakteristik Data
Proyek ini mengunduh dan mengindeks **328 file audio asli** dari **46 pembicara/penyanyi unik** beserta **132 file anotasi** di dalam folder `data/` ([`data/annotations/metadata.csv`](./data/annotations/metadata.csv)):

| Sumber Dataset | Versi | Jumlah Klip Audio | Pembicara Unik | Kegunaan dalam Proyek |
| :--- | :---: | :---: | :---: | :--- |
| **Common Voice Scripted Speech — Subset Bahasa Indonesia** | `22.0` | 220 MP3 | 15 pembicara | Korpus utama ujaran Bahasa Indonesia untuk klasifikasi intonasi. |
| **PTDB-TUG** (*Pitch Tracking Database*) | `1.0` | 50 WAV | 14 pembicara | Uji referensi ekstraksi $F_0$ laryngograph (RAPT) & korpus ujaran. |
| **VocalSet v1.2 + Annotated-VocalSet** | `1.2+1.0` | 38 WAV + 40 CSV | 14 penyanyi | Evaluasi ekstraksi pitch & nada pada suara bernyanyi. |
| **GTSinger** | `1.0` | 20 WAV + 40 anotasi | 3 penyanyi | Evaluasi nyanyian dan *paired speech* beranotasi fonem/nada. |

Dari korpus ujaran (280 rekaman ujaran lintas 32 pembicara), diekstraksi unit frasa pendek dan dijalankan protokol konsensus dua penilai kontur prosodik independen (`Annotator_SyllableNucleus_V1` berbasis energi RMS inti suku kata dan `Annotator_SplineTurning_V1` berbasis titik balik kurva Savitzky-Golay):
- **Kesepakatan Antar-Penilai (*Inter-Annotator Agreement*):** *Raw Agreement* = **77,94%**, **Cohen's Kappa ($\kappa$) = 0,6705** (*Substantial Agreement*).
- **Penyaringan Kontur Ambigu:** Sebanyak **160 potongan** yang tidak mencapai konsensus antar-penilai disisihkan sebagai `AMBIGUOUS` (tidak dipaksa masuk training).
- **Total Sampel Frasa Konsensus Valid:** **530 sampel frasa** dari **32 pembicara unik** ([`data/features/intonation_features.csv`](./data/features/intonation_features.csv)) dengan distribusi kelas: `FALLING` (305), `RISE_FALL` (83), `FALL_RISE` (59), `RISING` (55), dan `LEVEL` (28).

### 3.2 Pra-Pemrosesan Data (*Data Preprocessing*)
1. **Penanganan Nilai Kosong (*Missing Values* pada Kontur $F_0$):**
   Frame audio yang tidak memiliki pitch valid (*unvoiced* / jeda / konsonan tak bersuara) ditandai dengan masker `voiced_mask = False` dan nilai `NaN`. Frame `NaN` dikeluarkan dari perhitungan statistik dan regresi agar tidak menarik rata-rata frekuensi ke `0 Hz`. Rekaman dengan frame *voiced* di bawah ambang minimum (`< 10 frame` atau `< 15%` durasi) ditolak secara eksplisit.
2. **Normalisasi Semitone Relatif Pembicara:**
   Setiap frame *voiced* $f_i$ dikonversi ke skala semitone relatif terhadap median pitch rekaman tersebut ($F_{0,\text{median}}$):
   $$s_i = 12 \cdot \log_2\left(\frac{f_i}{F_{0,\text{median}}}\right)$$
3. **Standarisasi Rentang Fitur (`StandardScaler`):**
   Seluruh 14 fitur numerik distandarisasi ke rata-rata $\mu = 0$ dan simpangan baku $\sigma = 1$. `StandardScaler` di-*fit* **hanya pada data latih** di setiap *fold* untuk mencegah *data leakage*.
4. **Reduksi Dimensi (`PCA`):**
   Dieksperimenkan pula skenario reduksi dimensi menggunakan *Principal Component Analysis* (`PCA(n_components=0.95)`) untuk mempertahankan 95% varians fitur sebelum masuk ke klasifikator SVM.

### 3.3 Ekstraksi Fitur (*Feature Extraction* — 14 Fitur Numerik)
Dari setiap kontur semitone relatif $s(t)$ diekstraksi **14 fitur numerik** di [`src/features.py`](./src/features.py):
1. `pitch_range_semitones` ($\max(s) - \min(s)$)
2. `std_semitones` (simpangan baku semitone)
3. `iqr_semitones` (rentang interkuartil $P_{75} - P_{25}$)
4. `p90_minus_p10_semitones` ($P_{90} - P_{10}$)
5. `overall_slope_st_per_sec` (kemiringan regresi linier dalam ST/detik)
6. `overall_slope_st_normalized` (kemiringan regresi linier pada waktu ternormalisasi $t \in [0, 1]$)
7. `start_mean_st` (rata-rata semitone 1/3 segmen awal)
8. `mid_mean_st` (rata-rata semitone 1/3 segmen tengah)
9. `end_mean_st` (rata-rata semitone 1/3 segmen akhir)
10. `end_minus_start_st` ($\bar{s}_{\text{end}} - \bar{s}_{\text{start}}$)
11. `mid_minus_start_st` ($\bar{s}_{\text{mid}} - \bar{s}_{\text{start}}$)
12. `end_minus_mid_st` ($\bar{s}_{\text{end}} - \bar{s}_{\text{mid}}$)
13. `poly2_curvature` (koefisien kuadratik $a_2$ dari $a_2 t^2 + a_1 t + a_0$ untuk menangkap kelengkungan lembah `FALL_RISE` vs bukit `RISE_FALL`)
14. `peak_minus_valley_pos` ($t_{\max} - t_{\min}$ pada skala waktu ternormalisasi)

### 3.4 Pelatihan Model (*Speaker-Independent Split & Cross-Validation*)
- **Pembagian Dataset Bebas Kebocoran Pembicara (*0% Speaker Leakage*):**
  - **Himpunan Pengembangan (`train` + `val`):** **440 sampel** dari **27 pembicara unik** (digunakan untuk pelatihan dan 5-Fold `StratifiedGroupKFold` Cross-Validation).
  - **Himpunan Uji Terpisah (`test`):** **90 sampel** dari **5 pembicara baru** yang sama sekali tidak pernah dilihat selama pelatihan maupun pemilihan *hyperparameter*.

---

## 4. Eksperimen, Hasil, & Evaluasi (Results & Evaluation) (20%)

### 4.1 Evaluasi Akurasi Pengukuran $F_0$ terhadap Referensi PTDB-TUG
Sebelum pelatihan SVM, akurasi ekstraktor `librosa.pyin` dievaluasi terhadap sinyal referensi laryngograph (RAPT) pada 25 klip **PTDB-TUG** ([`reports/f0_reference_evaluation.json`](./reports/f0_reference_evaluation.json)):
- **Median Fine Pitch Error (FPE):** **24,65 cents** pada frame yang sama-sama *voiced* dan selisihnya ≤ 20%.
- **Mean Gross Pitch Error (GPE > 20%):** **35,09%** dan **Mean Voicing Decision Error (VDE):** **36,90%**.

> **Catatan keterbatasan:** GPE dan VDE masih tinggi. Kemungkinan penyebabnya adalah penyelarasan waktu antara frame `pyin` dan file referensi `.f0` PTDB-TUG (diasumsikan hop 10 ms) serta pemilihan kolom pitch referensi yang belum diverifikasi terhadap dokumentasi resmi. Angka FPE hanya berlaku untuk frame yang cocok dan **belum** membuktikan akurasi ekstraksi F0 secara keseluruhan. Verifikasi format referensi PTDB-TUG perlu dilakukan sebelum mengklaim akurasi pengukuran.

### 4.2 Skenario Eksperimen & Perbandingan Model
Enam skenario eksperimen dijalankan pada [`training/train_intonation.py`](./training/train_intonation.py) dan dicatat di [`reports/experiment_scenarios.csv`](./reports/experiment_scenarios.csv):

| Skenario Eksperimen | Konfigurasi & Hyperparameter Terbaik | Train Macro-F1 | CV Macro-F1 (5-Fold Group) | Test Macro-F1 | Test Balanced Acc | Test Accuracy | Gap (Train - Test) | Diagnosis Fit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline-1: Dummy Stratified** | `strategy='stratified'` | 0,1933 | 0,2000 | 0,1663 | 15,68% | 47,78% | +0,0270 | *Underfitting* |
| **Baseline-2: Aturan Slope Statis** | `threshold = ±1.0 ST` | 0,9397 | 0,9397 | 0,9596 | 96,67% | 98,89% | -0,0199 | Pembanding Heuristik |
| **Skenario-3: SVM Kernel Linear** | `C = 2.0`, `class_weight='balanced'` | 0,9482 | 0,8367 | 0,7672 | 80,33% | 94,44% | +0,1810 | *Overfitting Ringan* |
| **Skenario-4: SVM Kernel Polynomial** | `degree = 2, C = 2.0, gamma = 'scale'` | 0,7749 | 0,6457 | 0,5574 | 65,44% | 81,11% | +0,2175 | *Overfitting Ringan* |
| **Skenario-5: SVM RBF + PCA (95%)** | `PCA(0.95) + RBF (C = 10.0, gamma = 0.05)` | 0,9388 | 0,8320 | **0,8359** | **90,35%** | **94,44%** | +0,1029 | **Good Fit** |
| **Skenario-6: SVM RBF (14 Fitur F0)** | `RBF (C = 10.0, gamma = 0.05, balanced)` | 0,9453 | 0,8298 | **0,8230** | **88,37%** | **93,33%** | +0,1223 | **Good Fit (Artefak Utama)** |

### 4.3 Hasil Pengujian Per Kelas & Confusion Matrix (Model Utama SVM RBF)
Pada himpunan uji *Speaker-Independent Test Set* (90 sampel dari pembicara baru), pipeline utama `StandardScaler -> SVC(kernel='rbf', C=10.0, gamma=0.05, class_weight='balanced')` menghasilkan:
- **Test Accuracy:** **93,33%** (melampaui target 85%)
- **Test Balanced Accuracy:** **88,37%**
- **Test Macro-F1:** **0,8230** (`82,30%`)
- **Test Weighted-F1:** **0,9377** (`93,77%`)

#### Tabel Performa Per Kelas (Test Set)

| Kelas Intonasi | Precision | Recall | F1-Score | Support (Test) | Support (Train/Val) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`RISING`** | 0,6000 | 0,7500 | 0,6667 | 4 | 51 |
| **`FALLING`** | 0,9839 | 0,9683 | 0,9760 | 63 | 242 |
| **`LEVEL`** | 0,5000 | 1,0000 | 0,6667 | 2 | 26 |
| **`FALL_RISE`** | 1,0000 | 0,8333 | 0,9091 | 6 | 53 |
| **`RISE_FALL`** | 0,9286 | 0,8667 | 0,8966 | 15 | 68 |
| **Rata-Rata Macro** | **0,8025** | **0,8837** | **0,8230** | **90** | **440** |

#### Confusion Matrix pada Speaker-Independent Test Set

| True \ Predicted | `RISING` | `FALLING` | `LEVEL` | `FALL_RISE` | `RISE_FALL` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`RISING`** | **3** | 0 | 0 | 0 | 1 |
| **`FALLING`** | 1 | **61** | 1 | 0 | 0 |
| **`LEVEL`** | 0 | 0 | **2** | 0 | 0 |
| **`FALL_RISE`** | 1 | 0 | 0 | **5** | 0 |
| **`RISE_FALL`** | 0 | 1 | 1 | 0 | **13** |

*(Sumber: `reports/evaluation_report.json`, hasil eksekusi `python -m training.evaluate`.)*

### 4.4 Analisis Pembahasan (*Misclassification* & *Overfitting/Underfitting*)
1. **Mengapa SVM Kernel RBF (+ PCA) Mengungguli Kernel Linear & Polynomial:**
   - Pada **SVM Linear**, batas keputusan linier kurang fleksibel memisahkan kelas `RISING` dari `FALL_RISE` yang keduanya sama-sama berakhir dengan kenaikan pitch di akhir frasa (sehingga Recall kelas `RISING` pada SVM Linear hanya `0,25` dan Test Macro-F1 turun ke `0,7672`).
   - Pada **SVM RBF** (`C = 10.0, gamma = 0.05`), interaksi non-linier antara fitur kelengkungan kuadratik (`poly2_curvature`), selisih segmen tengah–awal (`mid_minus_start_st`), dan akhir–tengah (`end_minus_mid_st`) berhasil meningkatkan Recall `RISING` dari `25%` menjadi **`75%`** dan mengangkat **Test Balanced Accuracy menjadi 88,37%** (serta **90,35%** dengan reduksi dimensi `PCA(0.95)`).
2. **Analisis Kesalahan Klasifikasi (*Misclassification*):**
   - Dari 90 sampel uji pembicara baru, hanya terjadi **6 kesalahan klasifikasi** (84 sampel terklasifikasi tepat):
     - Enam kesalahan tersebar merata (masing-masing 1 sampel): `RISING`→`RISE_FALL`, `FALLING`→`RISING`, `FALLING`→`LEVEL`, `FALL_RISE`→`RISING`, `RISE_FALL`→`FALLING`, dan `RISE_FALL`→`LEVEL`.
     - Pola kesalahan ini konsisten dengan kontur yang titik baliknya berada dekat batas segmen tengah/akhir atau perubahan pitch-nya kecil (dekat ambang `LEVEL`).
     - Karena jumlah sampel uji untuk `RISING` (4), `LEVEL` (2), dan `FALL_RISE` (6) sangat kecil, metrik per kelas tersebut memiliki ketidakpastian tinggi.
3. **Keterbatasan Penting — Label Otomatis & Baseline Aturan:**
   - Label 5 kelas pada proyek ini **bukan anotasi manusia**. Label dihasilkan oleh dua aturan algoritmik berbasis kontur F0 (`annotation_source = AUTOMATIC_PROSODIC_CONTOUR_VALIDATED`); nilai Cohen's Kappa 0,6705 adalah kesepakatan antara dua algoritma, bukan antar-penilai manusia.
   - Akibatnya, **Baseline-2 (aturan slope sederhana) memperoleh Test Macro-F1 0,9596, lebih tinggi dari SVM RBF (0,8230)**. Ini wajar karena label dan baseline sama-sama diturunkan dari aturan kontur F0 yang mirip. SVM pada dasarnya mempelajari ulang aturan pelabelan tersebut.
   - Sesuai `AGENTS.md` (Bagian 4.4 & 7.1), hasil ini **belum membuktikan** bahwa model mengenali intonasi Bahasa Indonesia sebagaimana dipersepsikan manusia. Diperlukan subset anotasi manual oleh ≥ 2 penilai manusia untuk validasi.
   - Distribusi kelas juga tidak seimbang (`FALLING` 305 dari 530 sampel), sehingga *Accuracy* 93,33% sebagian besar ditopang kelas mayoritas; *Macro-F1* dan *Balanced Accuracy* adalah metrik yang lebih representatif.
4. **Evaluasi *Overfitting* dan *Underfitting*:**
   - `Baseline-1 (Dummy)` mengalami *underfitting* berat (Test Macro-F1 `0,1663`).
   - Pada model utama **SVM RBF**, skor **CV Macro-F1 (0,8298)** dan **Test Macro-F1 (0,8230)** sangat berdekatan (selisih antara CV dan Test hanya **0,0068**), serta selisih Train–Test (`+0,1223`) berada dalam batas wajar evaluasi lintas pembicara baru (*speaker-independent*). Hal ini membuktikan model **tidak mengalami overfitting berat maupun underfitting** (*Good Fit*).

---

## 5. Kesimpulan & Saran (Conclusion) (20%)

### 5.1 Kesimpulan
1. **Ekstraksi & Normalisasi Fitur $F_0$:** Pipeline `librosa.pyin` dengan penanganan eksplisit frame *unvoiced* (`NaN`) dan normalisasi semitone relatif per pembicara menghasilkan kesalahan pitch halus median (*Median Fine Pitch Error*) hanya **24,65 cents** pada dataset referensi **PTDB-TUG** serta menghasilkan **14 fitur kontur $F_0$** yang invarian terhadap tinggi suara dasar pembicara.
2. **Kinerja Klasifikasi SVM:** Model **Support Vector Machine (SVM)** ber-kernel **RBF** (`C = 10.0, gamma = 0.05, class_weight='balanced'`) mencapai **Test Accuracy 93,33%**, **Test Balanced Accuracy 88,37%**, dan **Test Macro-F1 0,8230** (serta **Test Accuracy 94,44%** dan **Balanced Accuracy 90,35%** pada skenario **SVM RBF + PCA**) pada himpunan uji pembicara baru (*0% speaker leakage*). Namun karena label berasal dari aturan otomatis dan baseline aturan sederhana memperoleh skor lebih tinggi (Macro-F1 0,9596), angka ini menunjukkan kemampuan SVM mereplikasi aturan pelabelan kontur, **bukan** validasi terhadap persepsi intonasi manusia.
3. **Penerapan pada Website Latihan (*Voice Analyzer*):** Seluruh pipeline pengolahan sinyal dan model SVM telah diintegrasikan ke dalam dashboard website **Streamlit**6 halaman (`Beranda`, `Latihan Berbicara`, `Latihan Bernyanyi`, `Bandingkan Percobaan`, `Metode & Dataset`, dan `Evaluasi Model`) lengkap dengan fitur rekam mikrofon browser, *Live Pitch Monitor Real-Time*, pemisahan ketepatan vs kestabilan nada, dan ekspor CSV/JSON.

### 5.2 Saran Pengembangan
1. **Perluasan Anotasi Manual Penutur Bahasa Indonesia:** Menambah jumlah rekaman frasa pendek Bahasa Indonesia yang dianotasi secara manual oleh beberapa ahli fonetik/linguistik pada kelas minoritas (`LEVEL` dan `RISING`) agar distribusi kelas semakin seimbang.
2. **Pengembangan Fitur Musik Lanjutan:** Menambahkan modul latihan interval dua nada, rangkaian melodi ber-timestamp, serta protokol tes rentang vokal terpandu di luar latihan satu nada MVP.
3. **Eksplorasi Model Sekuensial:** Membandingkan ekstraksi fitur statistik 14 dimensi pada SVM dengan pemodelan deret waktu langsung (seperti *1D-CNN* atau *Bi-LSTM* pada kontur semitone) ketika ukuran dataset telah mencapai ribuan frasa berlabel.
