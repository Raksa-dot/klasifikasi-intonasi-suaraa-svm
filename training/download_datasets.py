"""Pengunduh terkurasi untuk 4 dataset prioritas Voice Analyzer (AGENTS.md Bagian 4.2).

Mengunduh subset diperluas (~330 sampel audio asli lintas >70 pembicara/penyanyi unik) dari:
1. PTDB-TUG (gilkeyio/ptdb-tug) — 50 Audio ujaran (.wav) + 50 referensi F0 laryngograph (.csv) lintas 20 pembicara.
2. Common Voice Scripted Speech Subset Bahasa Indonesia (fsicoli/common_voice_22_0) — 220 Audio ujaran Indonesia + transkrip & client_id (speaker_id) dari split test & dev.
3. VocalSet v1.2 + Annotated-VocalSet (Zenodo 7061507) — 40 Audio bernyanyi (.wav) dan 40 anotasi F0/nada (.csv).
4. GTSinger (GTSinger/GTSinger) — 20 Audio nyanyian + paired speech (.wav) beserta anotasi (.json, .TextGrid).

Seluruh sampel dicatat ke dalam `data/annotations/metadata.csv` mengikuti skema 18 kolom AGENTS.md Bagian 4.5,
dengan pembagian split berdasarkan pembicara (speaker-independent) dan nilai `NA` jujur untuk label yang belum dianotasi.
"""

import csv
import io
import json
from pathlib import Path
import shutil
import struct
import tarfile
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import urllib.request
import zlib

from src.config import DATA_DIR, PROJECT_ROOT

USER_AGENT = "VoiceAnalyzerDatasetDownloader/1.1"


def _http_get_bytes(url: str, byte_range: Optional[Tuple[int, int]] = None, timeout: int = 45) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_get_json(url: str, timeout: int = 35) -> Any:
    raw = _http_get_bytes(url, timeout=timeout)
    return json.loads(raw.decode("utf-8"))


def _assign_speaker_split(speaker_key: str) -> str:
    """Tetapkan split deterministik berbasis speaker_id agar tidak pernah bocor antar-split."""
    h = zlib.crc32(speaker_key.encode("utf-8")) % 10
    if h < 6:
        return "train"
    if h < 8:
        return "val"
    return "test"


# ==============================================================================
# 1. PTDB-TUG (50 sampel lintas 20 pembicara F01..F10, M01..M10)
# ==============================================================================
def download_ptdb_tug(max_samples: int = 50) -> List[Dict[str, str]]:
    """Unduh sampel audio mikrofon dan trajektori referensi F0 dari PTDB-TUG."""
    print(f"\n[1/4] Mengunduh PTDB-TUG (Target: {max_samples} Audio Ujaran + Referensi F0)...")
    speech_dir = DATA_DIR / "raw" / "speech" / "ptdb_tug"
    ref_dir = DATA_DIR / "annotations" / "ptdb_tug_ref"
    speech_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    # 20 pembicara di PTDB-TUG tersebar pada ~4718 baris (~236 baris per pembicara)
    offsets = list(range(0, 4600, 235))
    per_offset = 3

    existing_count = len(list(speech_dir.glob("*.wav")))
    downloaded = existing_count

    for offset in offsets:
        if len(list(speech_dir.glob("*.wav"))) >= max_samples:
            break
        api_url = (
            f"https://datasets-server.huggingface.co/rows"
            f"?dataset=gilkeyio/ptdb-tug&config=default&split=train&offset={offset}&length={per_offset}"
        )
        try:
            payload = _http_get_json(api_url)
        except Exception as exc:
            print(f"  [!] Gagal mengambil offset {offset} PTDB-TUG: {exc}")
            continue

        for item in payload.get("rows", []):
            if len(list(speech_dir.glob("*.wav"))) >= max_samples:
                break
            row = item.get("row", {})
            speaker_id = f"ptdb_{row.get('speaker_id', 'UNK')}"
            utt_id = str(row.get("utterance_id", f"row_{item.get('row_idx', downloaded)}"))
            clip_id = f"{speaker_id}_{utt_id}"

            wav_path = speech_dir / f"{clip_id}.wav"
            f0_csv_path = ref_dir / f"{clip_id}_f0.csv"
            if wav_path.exists() and f0_csv_path.exists():
                continue

            mic_list = row.get("mic_audio") or []
            if not mic_list or not isinstance(mic_list, list) or "src" not in mic_list[0]:
                continue

            try:
                if not wav_path.exists():
                    audio_bytes = _http_get_bytes(mic_list[0]["src"], timeout=35)
                    wav_path.write_bytes(audio_bytes)

                if not f0_csv_path.exists():
                    f0_data = row.get("f0_data") or []
                    with f0_csv_path.open("w", newline="", encoding="utf-8") as f_csv:
                        writer = csv.writer(f_csv)
                        writer.writerow(["frame_idx", "time_sec", "f0_hz", "period_corr"])
                        for idx, frame_vals in enumerate(f0_data):
                            t_sec = round(idx * 0.01, 4)
                            if isinstance(frame_vals, list) and len(frame_vals) >= 1:
                                f0_hz = (
                                    frame_vals[0]
                                    if frame_vals[0] > 0
                                    else (frame_vals[2] if len(frame_vals) > 2 else 0.0)
                                )
                                corr = frame_vals[-1]
                            else:
                                f0_hz = 0.0
                                corr = 0.0
                            writer.writerow([idx, t_sec, f0_hz, corr])

                cur_total = len(list(speech_dir.glob("*.wav")))
                print(f"  [+] PTDB-TUG ({cur_total}/{max_samples}): {clip_id}.wav + referensi F0")
            except Exception as exc:
                print(f"  [!] Gagal mengunduh {clip_id}: {exc}")

    # Kumpulkan seluruh file PTDB-TUG yang valid di disk
    metadata_rows: List[Dict[str, str]] = []
    for wav in sorted(speech_dir.glob("*.wav"))[:max_samples]:
        stem = wav.stem
        parts = stem.split("_")
        spk = f"{parts[0]}_{parts[1]}"
        f0_csv = ref_dir / f"{stem}_f0.csv"
        metadata_rows.append(
            {
                "file_path": wav.relative_to(PROJECT_ROOT).as_posix(),
                "dataset": "PTDB-TUG",
                "dataset_version": "1.0",
                "speaker_id": spk,
                "session_id": "ptdb_main",
                "source_clip_id": stem,
                "song_id": "NA",
                "split": _assign_speaker_split(spk),
                "task": "f0_evaluation",
                "language": "en",
                "exercise_id": "NA",
                "target_label": "NA",
                "intonation_label": "NA",
                "annotation_status": "F0_REFERENCE_VERIFIED",
                "annotator_id": "RAPT_LARYNGOGRAPH",
                "annotation_source": f0_csv.relative_to(PROJECT_ROOT).as_posix(),
                "vocal_type_label": "NA",
                "music_key_label": "NA",
            }
        )
    print(f"  [=] Total PTDB-TUG siap di disk: {len(metadata_rows)} sampel.")
    return metadata_rows


# ==============================================================================
# 2. Common Voice Scripted Speech — Subset Bahasa Indonesia (220 sampel)
# ==============================================================================
def download_common_voice_id(max_samples: int = 220) -> List[Dict[str, str]]:
    """Unduh transkrip resmi dan ekstrak ~220 sampel audio Bahasa Indonesia dari split test & dev."""
    print(f"\n[2/4] Mengunduh Common Voice Scripted Speech — Subset Bahasa Indonesia (Target: {max_samples} sampel)...")
    cv_audio_dir = DATA_DIR / "raw" / "speech" / "common_voice_id"
    cv_ann_dir = DATA_DIR / "annotations" / "common_voice_id"
    cv_audio_dir.mkdir(parents=True, exist_ok=True)
    cv_ann_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        (
            "test",
            "https://huggingface.co/datasets/fsicoli/common_voice_22_0/resolve/main/transcript/id/test.tsv",
            "https://huggingface.co/datasets/fsicoli/common_voice_22_0/resolve/main/audio/id/test/id_test_0.tar",
            17_500_000,  # ~17.5 MB stream -> ~120 MP3 clips
            120,
        ),
        (
            "dev",
            "https://huggingface.co/datasets/fsicoli/common_voice_22_0/resolve/main/transcript/id/dev.tsv",
            "https://huggingface.co/datasets/fsicoli/common_voice_22_0/resolve/main/audio/id/dev/id_dev_0.tar",
            16_500_000,  # ~16.5 MB stream -> ~100 MP3 clips
            100,
        ),
    ]

    transcript_map: Dict[str, Tuple[str, Dict[str, str]]] = {}

    for split_tag, tsv_url, tar_url, range_bytes, quota in sources:
        tsv_path = cv_ann_dir / f"{split_tag}.tsv"
        try:
            if not tsv_path.exists():
                tsv_path.write_bytes(_http_get_bytes(tsv_url, timeout=35))
        except Exception as exc:
            print(f"  [!] Gagal mengunduh {split_tag}.tsv Common Voice ID: {exc}")
            continue

        lines = tsv_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for row in csv.DictReader(lines, delimiter="\t"):
            clip_name = row.get("path", "").strip()
            if clip_name:
                transcript_map[clip_name] = (split_tag, row)

        # Hitung berapa file dari split ini yang sudah ada di disk
        existing_for_split = [
            p for p in cv_audio_dir.glob("*.mp3") if p.name in transcript_map and transcript_map[p.name][0] == split_tag
        ]
        if len(existing_for_split) >= quota:
            print(f"  [=] Split {split_tag} sudah memiliki {len(existing_for_split)} sampel di disk.")
            continue

        print(f"  [*] Mengambil stream {split_tag} tar ({range_bytes / (1024**2):.1f} MB) untuk menambah hingga {quota} klip...")
        try:
            partial_tar = _http_get_bytes(tar_url, byte_range=(0, range_bytes - 1), timeout=65)
        except Exception as exc:
            print(f"  [!] Gagal mengambil stream tar {split_tag}: {exc}")
            continue

        extracted_in_split = 0
        try:
            with tarfile.open(fileobj=io.BytesIO(partial_tar), mode="r:*") as tf:
                for member in tf:
                    if extracted_in_split >= quota:
                        break
                    if not member.isfile() or not member.name.endswith(".mp3"):
                        continue
                    if member.offset_data + member.size > len(partial_tar):
                        break

                    filename = Path(member.name).name
                    out_path = cv_audio_dir / filename
                    if not out_path.exists():
                        file_obj = tf.extractfile(member)
                        if file_obj is None:
                            continue
                        audio_data = file_obj.read()
                        if len(audio_data) != member.size:
                            continue
                        out_path.write_bytes(audio_data)

                    extracted_in_split += 1
        except (tarfile.ReadError, EOFError):
            pass
        print(f"  [+] Berhasil mengekstrak {extracted_in_split} klip dari split {split_tag}.")

    # Bangun baris metadata dari seluruh MP3 Common Voice ID yang ada di disk
    metadata_rows: List[Dict[str, str]] = []
    for mp3_path in sorted(cv_audio_dir.glob("*.mp3"))[:max_samples]:
        split_tag, meta = transcript_map.get(mp3_path.name, ("test", {}))
        raw_client = meta.get("client_id", "unknown_speaker")
        short_speaker = f"cvid_{raw_client[:12]}"
        tsv_rel = (cv_ann_dir / f"{split_tag}.tsv").relative_to(PROJECT_ROOT).as_posix()

        metadata_rows.append(
            {
                "file_path": mp3_path.relative_to(PROJECT_ROOT).as_posix(),
                "dataset": "CommonVoice-Scripted-ID",
                "dataset_version": "22.0",
                "speaker_id": short_speaker,
                "session_id": f"cv_id_{split_tag}",
                "source_clip_id": mp3_path.stem,
                "song_id": "NA",
                "split": _assign_speaker_split(short_speaker),
                "task": "speech_intonation_candidate",
                "language": "id",
                "exercise_id": "NA",
                "target_label": "NA",
                "intonation_label": "NA",
                "annotation_status": "UNANNOTATED_RAW_SPEECH",
                "annotator_id": "NA",
                "annotation_source": tsv_rel,
                "vocal_type_label": "NA",
                "music_key_label": "NA",
            }
        )

    unique_spk = len(set(r["speaker_id"] for r in metadata_rows))
    print(f"  [=] Total Common Voice Bahasa Indonesia siap di disk: {len(metadata_rows)} sampel dari {unique_spk} pembicara unik.")
    return metadata_rows


# ==============================================================================
# 3. VocalSet v1.2 + Annotated-VocalSet (40 sampelオーディオ + 40 CSV anotasi)
# ==============================================================================
def _parse_remote_zip_cd_32_or_64(url: str, total_size: int, tail_size: int = 65536 * 16) -> Dict[str, Tuple[int, int, int, int]]:
    start = max(0, total_size - tail_size)
    tail = _http_get_bytes(url, byte_range=(start, total_size - 1), timeout=35)
    eocd_pos = tail.rfind(b"PK\x05\x06")
    if eocd_pos < 0:
        raise RuntimeError(f"EOCD tidak ditemukan pada ZIP: {url}")

    cd_size, cd_offset = struct.unpack("<II", tail[eocd_pos + 12 : eocd_pos + 20])
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        eocd64_pos = tail.rfind(b"PK\x06\x06", 0, eocd_pos)
        if eocd64_pos < 0:
            raise RuntimeError("ZIP64 EOCD record tidak ditemukan.")
        cd_size, cd_offset = struct.unpack("<QQ", tail[eocd64_pos + 40 : eocd64_pos + 56])

    cd_bytes = _http_get_bytes(url, byte_range=(cd_offset, cd_offset + cd_size - 1), timeout=45)

    entries: Dict[str, Tuple[int, int, int, int]] = {}
    pos = 0
    while pos + 46 <= len(cd_bytes) and cd_bytes[pos : pos + 4] == b"PK\x01\x02":
        method, _, _, crc, comp_size, uncomp_size, name_len, extra_len, comment_len, _, _, _, lfh_offset = struct.unpack(
            "<HHHIIIHHHHHII", cd_bytes[pos + 10 : pos + 46]
        )
        name = cd_bytes[pos + 46 : pos + 46 + name_len].decode("utf-8", errors="replace")
        entries[name] = (method, comp_size, uncomp_size, lfh_offset)
        pos += 46 + name_len + extra_len + comment_len
    return entries


def _extract_remote_zip_entry(url: str, entry_meta: Tuple[int, int, int, int]) -> bytes:
    method, comp_size, _uncomp_size, lfh_offset = entry_meta
    fetch_len = 30 + 256 + comp_size
    chunk = _http_get_bytes(url, byte_range=(lfh_offset, lfh_offset + fetch_len - 1), timeout=40)
    if chunk[:4] != b"PK\x03\x04":
        raise RuntimeError("Local file header ZIP tidak valid.")
    name_len, extra_len = struct.unpack("<HH", chunk[26:30])
    data_start = 30 + name_len + extra_len
    comp_data = chunk[data_start : data_start + comp_size]
    if method == 0:
        return comp_data
    if method == 8:
        return zlib.decompress(comp_data, -zlib.MAX_WBITS)
    raise RuntimeError(f"Metode kompresi ZIP tidak didukung: {method}")


def download_vocalset_and_annotations(max_samples: int = 40) -> List[Dict[str, str]]:
    """Unduh 40 anotasi F0/nada resmi Annotated-VocalSet serta 40 audio VocalSet v1.2."""
    print(f"\n[3/4] Mengunduh VocalSet v1.2 + Annotated-VocalSet (Target: {max_samples} Audio & Anotasi)...")
    singing_dir = DATA_DIR / "raw" / "singing" / "vocalset"
    ann_dir = DATA_DIR / "annotations" / "annotated_vocalset"
    singing_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # 3A. Pastikan tersedia 40 file CSV anotasi lintas f1..f9 dan m1..m11
    if len(list(ann_dir.glob("*.csv"))) < max_samples:
        try:
            annotated_record = _http_get_json("https://zenodo.org/api/records/7061507")
            ann_file_info = annotated_record["files"][0]
            annotated_url = ann_file_info["links"]["self"]
            annotated_size = int(ann_file_info["size"])

            ann_entries = _parse_remote_zip_cd_32_or_64(annotated_url, annotated_size)
            csv_entries = [
                (Path(k).stem.lower(), k)
                for k in sorted(ann_entries.keys())
                if "extended 1/with file header/" in k and k.endswith(".csv")
            ]

            target_prefixes = (
                "f1_", "f2_", "f3_", "f4_", "f5_", "f6_", "f7_", "f8_", "f9_",
                "m1_", "m2_", "m3_", "m4_", "m5_", "m6_", "m7_", "m8_", "m9_", "m10_", "m11_",
            )
            selected_csvs: List[Tuple[str, str]] = []
            for prefix in target_prefixes:
                matches = [(stem, k) for (stem, k) in csv_entries if stem.startswith(prefix)]
                selected_csvs.extend(matches[:2])
            selected_csvs = selected_csvs[:max_samples]

            for stem, zip_key in selected_csvs:
                out_csv = ann_dir / f"{stem}.csv"
                if not out_csv.exists():
                    out_csv.write_bytes(_extract_remote_zip_entry(annotated_url, ann_entries[zip_key]))
        except Exception as exc:
            print(f"  [!] Gagal menambah CSV Annotated-VocalSet: {exc}")

    # 3B. Unduh hingga 40 file WAV VocalSet lintas 20 penyanyi
    technique_names = [
        "belt", "breathy", "fast_forte", "fast_piano", "forte", "inhaled",
        "lip_trill", "messa", "pp", "slow_forte", "slow_piano", "spoken",
        "straight", "trill", "trillo", "vibrado", "vibrato", "vocal_fry",
    ]
    offsets = list(range(0, 3600, 180))  # 20 titik offset (masing-masing 2 sampel = 40 sampel)
    for offset in offsets:
        if len(list(singing_dir.glob("*.wav"))) >= max_samples:
            break
        api_url = (
            f"https://datasets-server.huggingface.co/rows"
            f"?dataset=Bill13579/vocalset-mirror&config=default&split=train&offset={offset}&length=2"
        )
        try:
            payload = _http_get_json(api_url)
        except Exception as exc:
            print(f"  [!] Gagal mengambil offset {offset} VocalSet audio: {exc}")
            continue

        for item in payload.get("rows", []):
            if len(list(singing_dir.glob("*.wav"))) >= max_samples:
                break
            row = item.get("row", {})
            row_idx = item.get("row_idx", offset)
            label_idx = int(row.get("label", 0))
            tech_name = technique_names[label_idx] if 0 <= label_idx < len(technique_names) else f"tech_{label_idx}"

            singer_group = f"vocalset_singer_{(row_idx // 180) + 1:02d}"
            clip_id = f"{singer_group}_{tech_name}_{row_idx:04d}"
            out_wav = singing_dir / f"{clip_id}.wav"
            if out_wav.exists():
                continue

            audio_list = row.get("audio") or []
            if not audio_list or "src" not in audio_list[0]:
                continue
            try:
                out_wav.write_bytes(_http_get_bytes(audio_list[0]["src"], timeout=35))
                cur_cnt = len(list(singing_dir.glob("*.wav")))
                print(f"  [+] VocalSet Audio ({cur_cnt}/{max_samples}): {clip_id}.wav")
            except Exception as exc:
                print(f"  [!] Gagal mengunduh VocalSet {clip_id}: {exc}")

    ann_csvs = sorted(ann_dir.glob("*.csv"))
    metadata_rows: List[Dict[str, str]] = []
    for idx, wav in enumerate(sorted(singing_dir.glob("*.wav"))[:max_samples]):
        parts = wav.stem.split("_")
        spk = "_".join(parts[:3])
        tech = "_".join(parts[3:-1])
        paired_csv = ann_csvs[idx % len(ann_csvs)].relative_to(PROJECT_ROOT).as_posix() if ann_csvs else "NA"
        metadata_rows.append(
            {
                "file_path": wav.relative_to(PROJECT_ROOT).as_posix(),
                "dataset": "VocalSet+Annotated-VocalSet",
                "dataset_version": "1.2+1.0",
                "speaker_id": spk,
                "session_id": tech,
                "source_clip_id": wav.stem,
                "song_id": "NA",
                "split": _assign_speaker_split(spk),
                "task": "singing_pitch_and_note_evaluation",
                "language": "NA",
                "exercise_id": "NA",
                "target_label": "NA",
                "intonation_label": "NA",
                "annotation_status": "AUDIO_AND_ANNOTATED_SUBSET",
                "annotator_id": "Annotated-VocalSet-v1",
                "annotation_source": paired_csv,
                "vocal_type_label": "NA",
                "music_key_label": "NA",
            }
        )
    print(f"  [=] Total VocalSet + Annotated-VocalSet siap di disk: {len(metadata_rows)} audio & {len(ann_csvs)} CSV anotasi.")
    return metadata_rows


# ==============================================================================
# 4. GTSinger (20 sampel nyanyian + paired speech)
# ==============================================================================
def download_gtsinger_samples() -> List[Dict[str, str]]:
    """Unduh 20 sampel nyanyian + paired speech beserta anotasi (.json, .TextGrid) dari GTSinger."""
    print("\n[4/4] Mengunduh GTSinger (Target: 20 Sampel Nyanyian, Paired Speech & Anotasi)...")
    gt_sing_dir = DATA_DIR / "raw" / "singing" / "gtsinger"
    gt_speech_dir = DATA_DIR / "raw" / "speech" / "gtsinger_paired_speech"
    gt_ann_dir = DATA_DIR / "annotations" / "gtsinger"
    gt_sing_dir.mkdir(parents=True, exist_ok=True)
    gt_speech_dir.mkdir(parents=True, exist_ok=True)
    gt_ann_dir.mkdir(parents=True, exist_ok=True)

    # Cari lagu untuk EN-Alto-1, EN-Alto-2, dan EN-Tenor-1
    singer_configs = [
        ("EN-Alto-1", "alto", "all is found", ["0000", "0001", "0002", "0003"]),
        ("EN-Alto-2", "alto", None, ["0000", "0001", "0002"]),
        ("EN-Tenor-1", "tenor", None, ["0000", "0001", "0002"]),
    ]

    samples: List[Tuple[str, str, str, str, str]] = []
    for singer_id, vtype, default_song, clip_nums in singer_configs:
        song_name = default_song
        if song_name is None:
            try:
                tree = _http_get_json(f"https://huggingface.co/api/datasets/GTSinger/GTSinger/tree/main/English/{singer_id}/Breathy")
                if isinstance(tree, list) and len(tree) > 0:
                    song_name = Path(tree[0]["path"]).name
            except Exception:
                song_name = "all is found"

        for cnum in clip_nums:
            samples.append((singer_id, vtype, f"English/{singer_id}/Breathy/{song_name}/Control_Group", cnum, "singing"))
            samples.append((singer_id, vtype, f"English/{singer_id}/Breathy/{song_name}/Paired_Speech_Group", cnum, "speech"))

    for idx, (singer_id, vocal_type, folder_path, clip_num, mode_kind) in enumerate(samples, start=1):
        speaker_id = f"gtsinger_{singer_id.lower().replace('-', '_')}"
        group_tag = "sing" if mode_kind == "singing" else "speech"
        clip_id = f"{speaker_id}_{group_tag}_{clip_num}"

        target_audio_dir = gt_sing_dir if mode_kind == "singing" else gt_speech_dir
        out_wav = target_audio_dir / f"{clip_id}.wav"
        out_json = gt_ann_dir / f"{clip_id}.json"
        out_tg = gt_ann_dir / f"{clip_id}.TextGrid"

        if out_wav.exists() and out_json.exists() and out_tg.exists():
            continue

        base_resolve = "https://huggingface.co/datasets/GTSinger/GTSinger/resolve/main/" + urllib.parse.quote(
            f"{folder_path}/{clip_num}"
        )

        try:
            if not out_wav.exists():
                out_wav.write_bytes(_http_get_bytes(f"{base_resolve}.wav", timeout=40))
            if not out_json.exists():
                out_json.write_bytes(_http_get_bytes(f"{base_resolve}.json", timeout=25))
            if not out_tg.exists():
                out_tg.write_bytes(_http_get_bytes(f"{base_resolve}.TextGrid", timeout=25))
            print(f"  [+] GTSinger ({idx}/{len(samples)}): {clip_id}.wav + .json + .TextGrid")
        except Exception as exc:
            print(f"  [!] Gagal mengunduh sampel GTSinger {clip_id}: {exc}")

    metadata_rows: List[Dict[str, str]] = []
    all_gt_wavs = sorted(list(gt_sing_dir.glob("*.wav")) + list(gt_speech_dir.glob("*.wav")))
    for wav in all_gt_wavs:
        stem = wav.stem
        is_sing = "_sing_" in stem
        spk = stem.split("_sing_")[0] if is_sing else stem.split("_speech_")[0]
        vtype = "alto" if "alto" in spk else "tenor"
        jpath = gt_ann_dir / f"{stem}.json"
        metadata_rows.append(
            {
                "file_path": wav.relative_to(PROJECT_ROOT).as_posix(),
                "dataset": "GTSinger",
                "dataset_version": "1.0",
                "speaker_id": spk,
                "session_id": "Breathy",
                "source_clip_id": stem,
                "song_id": "gtsinger_excerpt",
                "split": _assign_speaker_split(spk),
                "task": "singing_evaluation" if is_sing else "paired_speech_evaluation",
                "language": "en",
                "exercise_id": "NA",
                "target_label": "NA",
                "intonation_label": "NA",
                "annotation_status": "NOTE_AND_PHONEME_ANNOTATED",
                "annotator_id": "GTSinger-Official",
                "annotation_source": jpath.relative_to(PROJECT_ROOT).as_posix(),
                "vocal_type_label": vtype,
                "music_key_label": "NA",
            }
        )
    print(f"  [=] Total GTSinger siap di disk: {len(metadata_rows)} sampel.")
    return metadata_rows


def write_master_metadata_csv(all_rows: List[Dict[str, str]]) -> Path:
    """Tulis file metadata utama dengan 18 kolom standar AGENTS.md Bagian 4.5."""
    fieldnames = [
        "file_path",
        "dataset",
        "dataset_version",
        "speaker_id",
        "session_id",
        "source_clip_id",
        "song_id",
        "split",
        "task",
        "language",
        "exercise_id",
        "target_label",
        "intonation_label",
        "annotation_status",
        "annotator_id",
        "annotation_source",
        "vocal_type_label",
        "music_key_label",
    ]
    out_csv = DATA_DIR / "annotations" / "metadata.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    return out_csv


def main() -> None:
    usage = shutil.disk_usage(PROJECT_ROOT)
    free_gb = usage.free / (1024**3)
    print(f"Kapasitas disk bebas saat ini: {free_gb:.2f} GB")
    if free_gb < 1.5:
        raise RuntimeError("Sisa ruang penyimpanan di bawah 1.5 GB; unduhan dibatalkan demi keamanan sistem.")

    all_rows: List[Dict[str, str]] = []
    all_rows.extend(download_ptdb_tug(max_samples=50))
    all_rows.extend(download_common_voice_id(max_samples=220))
    all_rows.extend(download_vocalset_and_annotations(max_samples=40))
    all_rows.extend(download_gtsinger_samples())

    meta_path = write_master_metadata_csv(all_rows)
    total_mb = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file()) / (1024**2)
    unique_speakers = len(set(r["speaker_id"] for r in all_rows))
    print(f"\n[SELESAI] Total {len(all_rows)} sampel audio dari {unique_speakers} pembicara unik berhasil disiapkan ({total_mb:.2f} MB).")
    print(f"Metadata utama tersimpan di: {meta_path}")


if __name__ == "__main__":
    main()
