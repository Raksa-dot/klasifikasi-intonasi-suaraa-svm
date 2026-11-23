# Direktori Data (`data/`) — Status Dataset Terunduh

Direktori ini menyimpan subset diperluas yang telah diverifikasi dari **4 dataset prioritas** pada [`../AGENTS.md`](../AGENTS.md) Bagian 4.2. Ukuran total di disk adalah **91,45 MB** (aman untuk kapasitas penyimpanan lokal dan folder OneDrive), mencakup **328 file audio asli** dari **46 pembicara/penyanyi unik** beserta **132 file anotasi/referensi**.

Untuk menjalankan ulang pemeriksaan atau pengunduhan dataset:

```powershell
python -m training.download_datasets
```

---

## 1. Ringkasan Dataset yang Telah Diunduh (328 Sampel Audio)

| Dataset | Versi | Jumlah Klip Audio | Pembicara / Penyanyi Unik | Split (`train` / `val` / `test`) | Lokasi Audio Mentah (`data/raw/`) | Lokasi Anotasi (`data/annotations/`) |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Common Voice Scripted Speech — Subset Bahasa Indonesia** | `22.0` | **220 MP3** | 15 pembicara (`cvid_*`) | `128 / 47 / 45` | `data/raw/speech/common_voice_id/` | `data/annotations/common_voice_id/test.tsv` & `dev.tsv` |
| **PTDB-TUG** | `1.0` | **50 WAV** | 14 pembicara (`F01`–`F10`, `M01`–`M07`) | `28 / 11 / 11` | `data/raw/speech/ptdb_tug/` | `data/annotations/ptdb_tug_ref/` (50 file `*_f0.csv`) |
| **VocalSet v1.2 + Annotated-VocalSet** | `1.2 + 1.0` | **38 WAV** | 14 penyanyi (`vocalset_singer_*`) | `22 / 8 / 8` | `data/raw/singing/vocalset/` | `data/annotations/annotated_vocalset/` (40 file `*.csv`) |
| **GTSinger** | `1.0` | **20 WAV** (10 nyanyian + 10 *paired speech*) | 3 penyanyi (`EN-Alto-1`, `EN-Alto-2`, `EN-Tenor-1`) | `8 / 6 / 6` | `data/raw/singing/gtsinger/` & `data/raw/speech/gtsinger_paired_speech/` | `data/annotations/gtsinger/` (20 `*.json` + 20 `*.TextGrid`) |
| **Total Keseluruhan** | — | **328 Audio** | **46 Pembicara Unik** | **`186 / 72 / 70`** | — | [`data/annotations/metadata.csv`](./annotations/metadata.csv) |

---

## 2. Pembagian Split Berbasis Pembicara (Speaker-Independent Split)

Seluruh **328 sampel audio** diindeks di [`data/annotations/metadata.csv`](./annotations/metadata.csv) dengan pembagian proporsional per dataset berbasis `speaker_id` (~57% `train`, ~22% `val`, ~21% `test`) dan **0% kebocoran pembicara lintas split**:

- **`train`:** 186 sampel
- **`val`:** 72 sampel
- **`test`:** 70 sampel

---

## 3. Skema Kolom `data/annotations/metadata.csv`

Sesuai [`../AGENTS.md`](../AGENTS.md) Bagian 4.5, file [`data/annotations/metadata.csv`](./annotations/metadata.csv) menggunakan 18 kolom standar:

```csv
file_path,dataset,dataset_version,speaker_id,session_id,source_clip_id,song_id,split,task,language,exercise_id,target_label,intonation_label,annotation_status,annotator_id,annotation_source,vocal_type_label,music_key_label
```

- Nilai label yang belum tersedia (seperti `intonation_label` lima kelas pada audio mentah Common Voice sebelum dianotasi) diisi secara jujur dengan **`NA`**, bukan ditebak dari tanda baca.
- File audio mentah di dalam `data/raw/` dilindungi oleh `.gitignore` agar tidak ter-commit ke repositori Git.
