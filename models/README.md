# Direktori Artefak Model (`models/`)

Direktori ini menyimpan artefak model Machine Learning yang telah dilatih dan divalidasi secara offline beserta file metadatanya, sesuai [`../AGENTS.md`](../AGENTS.md) Bagian 7.

## Artefak yang Ditargetkan

- `intonation_svm.joblib` — Pipeline `StandardScaler -> SVC(kernel="rbf", class_weight="balanced")` untuk klasifikasi 5 pola intonasi ujaran (`RISING`, `FALLING`, `LEVEL`, `FALL_RISE`, `RISE_FALL`) setelah lolos evaluasi *speaker-independent*.
- `intonation_svm_metadata.json` — Metadata pendamping wajib yang memuat:
  - Versi model & tanggal training
  - Definisi tugas & daftar kelas
  - Daftar nama fitur beserta urutannya
  - Versi feature extractor & parameter audio (`sample_rate`, `fmin`, `fmax`, `frame_length`, `hop_length`)
  - Versi Python, `scikit-learn`, dan `librosa`
  - Sumber/versi data, metode split (`speaker-independent`), dan ringkasan metrik evaluasi
- `vocal_range_svm.joblib` *(Opsional / Fitur Lanjutan)*
- `music_key_svm.joblib` *(Opsional / Fitur Lanjutan)*

## Status Saat Ini

Belum ada artefak model SVM yang diaktifkan (`MODEL_NOT_READY`). Website tetap menjalankan seluruh fitur latihan berbasis pengukuran sinyal ($F_0$, konversi nada, selisih *cents*, kestabilan nada, dan perbandingan percobaan) secara jujur tanpa prediksi buatan.
