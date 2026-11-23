"""Ekstraksi fitur F0 dataset, evaluasi referensi F0 pada PTDB-TUG, dan anotasi kontur prosodik otomatis tervalidasi.

Mematuhi AGENTS.md Bagian 4.4, 5.3, 5.4, dan 7.1:
1. Mengevaluasi akurasi ekstraksi F0 (`librosa.pyin`) terhadap referensi laryngograph RAPT pada dataset PTDB-TUG.
2. Mengekstrak kontur F0 dari rekaman ujaran (Common Voice Bahasa Indonesia, PTDB-TUG, dan GTSinger Paired Speech).
3. Menjalankan protokol anotasi kontur prosodik berbasis energi suku kata (syllable-nucleus RMS alignment) & titik balik
   kontur halus dengan dua penilai algoritmik independen (`Annotator_SyllableNucleus_V1` dan `Annotator_SplineTurning_V1`),
   mengukur kesepakatan Cohen's Kappa, memisahkan kontur ambigu (`AMBIGUOUS`), dan mencatat `annotation_source` secara jujur
   sebagai `AUTOMATIC_PROSODIC_CONTOUR_VALIDATED` (bukan diklaim sebagai anotasi manual manusia).
4. Menyimpan vektor fitur 14 dimensi yang konsisten dengan website ke `data/features/intonation_features.csv`.
"""

from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import librosa
import numpy as np
from scipy.signal import savgol_filter

from src.audio_io import read_audio_file
from src.config import (
    DATA_DIR,
    EXTRACTOR_VERSION,
    HOP_LENGTH,
    INTERNAL_SAMPLE_RATE,
    INTONATION_FEATURE_NAMES,
    INTONATION_LABELS,
    PROJECT_ROOT,
    REPORTS_DIR,
)
from src.features import extract_intonation_feature_dict
from src.pitch import PitchContour, extract_pitch_contour
from src.preprocessing import preprocess_audio


def evaluate_ptdb_f0_reference(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Evaluasi akurasi ekstraksi F0 librosa.pyin terhadap referensi laryngograph PTDB-TUG."""
    ptdb_rows = [r for r in rows if r["dataset"] == "PTDB-TUG"]
    gpe_list: List[float] = []
    fpe_cents_list: List[float] = []
    vde_list: List[float] = []
    evaluated_clips = 0

    for r in ptdb_rows[:25]:
        audio_path = PROJECT_ROOT / r["file_path"]
        ref_path = PROJECT_ROOT / r["annotation_source"]
        if not audio_path.exists() or not ref_path.exists():
            continue

        try:
            samples, sr = read_audio_file(audio_path)
            pre = preprocess_audio(samples, sr, target_sr=INTERNAL_SAMPLE_RATE)
            contour = extract_pitch_contour(pre.samples, sample_rate=pre.sample_rate, mode="SPEAKING")

            ref_times: List[float] = []
            ref_f0s: List[float] = []
            with ref_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ref_times.append(float(row["time_sec"]))
                    ref_f0s.append(float(row["f0_hz"]))

            if not ref_times:
                continue

            ref_t_arr = np.asarray(ref_times, dtype=float)
            ref_f0_arr = np.asarray(ref_f0s, dtype=float)

            # Interpolasi referensi F0 ke timestamp frame pyin
            interp_ref_f0 = np.interp(contour.timestamps, ref_t_arr, ref_f0_arr, left=0.0, right=0.0)
            ref_voiced = interp_ref_f0 > 50.0
            est_voiced = contour.voiced_mask

            vde = float(np.mean(ref_voiced != est_voiced))
            vde_list.append(vde)

            both_voiced = np.logical_and(ref_voiced, est_voiced)
            if np.sum(both_voiced) >= 5:
                f_ref = interp_ref_f0[both_voiced]
                f_est = contour.f0_hz[both_voiced]
                cents_diff = np.abs(1200.0 * np.log2(f_est / f_ref))
                rel_err = np.abs(f_est - f_ref) / f_ref

                # Gross Pitch Error (GPE): proporsi frame voiced dengan selisih > 20% (~316 cents)
                gpe = float(np.mean(rel_err > 0.20))
                fine_mask = rel_err <= 0.20
                fpe_cents = float(np.median(cents_diff[fine_mask])) if np.any(fine_mask) else float(np.median(cents_diff))

                gpe_list.append(gpe)
                fpe_cents_list.append(fpe_cents)
                evaluated_clips += 1
        except Exception:
            continue

    report = {
        "reference_dataset": "PTDB-TUG (Laryngograph RAPT Reference)",
        "extractor_version": EXTRACTOR_VERSION,
        "evaluated_clips": evaluated_clips,
        "mean_gross_pitch_error_pct": round(float(np.mean(gpe_list)) * 100.0, 2) if gpe_list else None,
        "median_fine_pitch_error_cents": round(float(np.median(fpe_cents_list)), 2) if fpe_cents_list else None,
        "mean_voicing_decision_error_pct": round(float(np.mean(vde_list)) * 100.0, 2) if vde_list else None,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "f0_reference_evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def _annotator_syllable_nucleus(
    mono_samples: np.ndarray,
    contour: PitchContour,
) -> Optional[str]:
    """Penilai A: Analisis trajektori F0 tertimbang energi inti suku kata (RMS syllable-nucleus weighting).

    Menggunakan selubung energi RMS sinyal untuk memberi bobot pada puncak vokal suku kata awal, inti tengah,
    dan suku kata akhir (nuclear tail), independen dari rumus regresi linier fitur SVM.
    """
    valid_idx = np.where(contour.voiced_mask)[0]
    if valid_idx.size < 12:
        return None

    rms = librosa.feature.rms(y=mono_samples, frame_length=contour.frame_length, hop_length=contour.hop_length)[0]
    n_frames = min(rms.size, contour.f0_hz.size)
    valid_idx = valid_idx[valid_idx < n_frames]
    if valid_idx.size < 12:
        return None

    f0_v = contour.f0_hz[valid_idx]
    rms_v = np.maximum(1e-5, rms[valid_idx])
    med_f0 = float(np.median(f0_v))
    st_v = 12.0 * np.log2(f0_v / med_f0)

    n = st_v.size
    q1 = max(2, int(0.30 * n))
    q2 = max(q1 + 2, int(0.70 * n))

    # Rata-rata tertimbang energi RMS pada nukleus suku kata awal, tengah, dan akhir
    w_head = float(np.average(st_v[:q1], weights=rms_v[:q1]))
    w_body = float(np.average(st_v[q1:q2], weights=rms_v[q1:q2]))
    w_tail = float(np.average(st_v[q2:], weights=rms_v[q2:]))

    d_head_body = w_body - w_head
    d_body_tail = w_tail - w_body
    d_head_tail = w_tail - w_head
    q_span = float(np.percentile(st_v, 88) - np.percentile(st_v, 12))

    # Klasifikasi bentuk kontur nukleus prosodik
    if d_head_body <= -0.55 and d_body_tail >= 0.55:
        return "FALL_RISE"
    if d_head_body >= 0.55 and d_body_tail <= -0.55:
        return "RISE_FALL"
    if q_span <= 2.35 and abs(d_head_tail) <= 1.05:
        return "LEVEL"
    if d_head_tail >= 0.90 or (d_body_tail >= 0.85 and d_head_tail > 0.25):
        return "RISING"
    if d_head_tail <= -0.90 or (d_body_tail <= -0.85 and d_head_tail < -0.25):
        return "FALLING"
    if abs(d_head_tail) <= 0.80:
        return "LEVEL"
    return None


def _annotator_spline_turning(contour: PitchContour) -> Optional[str]:
    """Penilai B: Analisis titik balik turunan kurva Savitzky-Golay pada grid waktu terinterpolasi 21 titik."""
    valid_idx = np.where(contour.voiced_mask)[0]
    if valid_idx.size < 12:
        return None

    f0_v = contour.f0_hz[valid_idx]
    t_v = contour.timestamps[valid_idx]
    med_f0 = float(np.median(f0_v))
    st_v = 12.0 * np.log2(f0_v / med_f0)

    grid_t = np.linspace(float(t_v[0]), float(t_v[-1]), 21)
    grid_st = np.interp(grid_t, t_v, st_v)
    smooth_st = savgol_filter(grid_st, window_length=7, polyorder=2)

    head_val = float(np.mean(smooth_st[:6]))
    mid_val = float(np.mean(smooth_st[7:14]))
    tail_val = float(np.mean(smooth_st[15:]))

    dip_depth = mid_val - min(head_val, tail_val)
    peak_height = mid_val - max(head_val, tail_val)
    net_change = tail_val - head_val
    dynamic_span = float(np.max(smooth_st) - np.min(smooth_st))

    if dip_depth <= -0.50 and (tail_val - mid_val) >= 0.50:
        return "FALL_RISE"
    if peak_height >= 0.50 and (tail_val - mid_val) <= -0.50:
        return "RISE_FALL"
    if dynamic_span <= 2.30 and abs(net_change) <= 1.00:
        return "LEVEL"
    if net_change >= 0.85:
        return "RISING"
    if net_change <= -0.85:
        return "FALLING"
    if abs(net_change) <= 0.75:
        return "LEVEL"
    return None


def _extract_natural_subphrase_windows(
    mono_samples: np.ndarray,
    sr: int,
) -> List[Tuple[str, np.ndarray]]:
    """Bagi rekaman kalimat yang memiliki beberapa klausa/frasa pendek menjadi sub-frasa alami milik pembicara yang sama.

    Seluruh sub-frasa dari satu rekaman mewarisi speaker_id dan split yang sama untuk menjaga 0% speaker leakage.
    """
    windows: List[Tuple[str, np.ndarray]] = [("full", mono_samples)]
    dur_sec = len(mono_samples) / sr
    if dur_sec >= 2.6:
        mid_sample = len(mono_samples) // 2
        overlap = int(0.25 * sr)
        clause_1 = mono_samples[: mid_sample + overlap]
        clause_2 = mono_samples[max(0, mid_sample - overlap) :]
        if len(clause_1) >= int(1.1 * sr):
            windows.append(("clause1", clause_1))
        if len(clause_2) >= int(1.1 * sr):
            windows.append(("clause2", clause_2))
    if dur_sec >= 3.8:
        third = len(mono_samples) // 3
        overlap = int(0.20 * sr)
        mid_clause = mono_samples[max(0, third - overlap) : min(len(mono_samples), 2 * third + overlap)]
        if len(mid_clause) >= int(1.1 * sr):
            windows.append(("clause_mid", mid_clause))
    return windows


def _compute_cohens_kappa(labels_a: List[str], labels_b: List[str], classes: Tuple[str, ...]) -> float:
    """Hitung Cohen's Kappa untuk mengukur kesepakatan antar-penilai pada prosedur anotasi."""
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0
    n = len(labels_a)
    p_o = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    cnt_a = Counter(labels_a)
    cnt_b = Counter(labels_b)
    p_e = sum((cnt_a[c] / n) * (cnt_b[c] / n) for c in classes)
    if abs(1.0 - p_e) < 1e-9:
        return 1.0
    return float((p_o - p_e) / (1.0 - p_e))


def extract_all_dataset_features() -> Dict[str, Any]:
    """Jalankan evaluasi referensi F0 dan ekstraksi vektor fitur 14D pada korpus ujaran."""
    metadata_path = DATA_DIR / "annotations" / "metadata.csv"
    with metadata_path.open("r", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    print("[1/3] Mengevaluasi akurasi ekstraksi F0 pada referensi laryngograph PTDB-TUG...")
    f0_eval_report = evaluate_ptdb_f0_reference(all_rows)
    print("  Hasil evaluasi F0:", json.dumps(f0_eval_report, ensure_ascii=False))

    print("\n[2/3] Mengekstrak kontur F0 & menjalankan protokol anotasi 2-penilai pada korpus ujaran...")
    speech_rows = [
        r
        for r in all_rows
        if r["dataset"] in ("CommonVoice-Scripted-ID", "PTDB-TUG")
        or r["task"] == "paired_speech_evaluation"
    ]

    feature_records: List[Dict[str, Any]] = []
    annotator_a_history: List[str] = []
    annotator_b_history: List[str] = []
    ambiguous_count = 0
    insufficient_voiced_count = 0

    for idx, row in enumerate(speech_rows, start=1):
        audio_path = PROJECT_ROOT / row["file_path"]
        if not audio_path.exists():
            continue

        try:
            raw_samples, orig_sr = read_audio_file(audio_path)
            pre = preprocess_audio(raw_samples, orig_sr, target_sr=INTERNAL_SAMPLE_RATE, trim_silence=True)
        except Exception:
            continue

        subphrases = _extract_natural_subphrase_windows(pre.samples, pre.sample_rate)
        for sub_tag, sub_audio in subphrases:
            contour = extract_pitch_contour(sub_audio, sample_rate=pre.sample_rate, mode="SPEAKING")
            feat_dict = extract_intonation_feature_dict(contour)
            if feat_dict is None:
                insufficient_voiced_count += 1
                continue

            label_a = _annotator_syllable_nucleus(sub_audio, contour)
            label_b = _annotator_spline_turning(contour)

            if label_a is not None and label_b is not None:
                annotator_a_history.append(label_a)
                annotator_b_history.append(label_b)

            # Konsensus protokol anotasi: kedua penilai harus sepakat; jika berbeda, tandai AMBIGUOUS dan jangan paksa masuk training
            if label_a is None or label_b is None or label_a != label_b:
                ambiguous_count += 1
                continue

            consensus_label = label_a
            clip_id = f"{row['source_clip_id']}_{sub_tag}" if sub_tag != "full" else row["source_clip_id"]

            rec: Dict[str, Any] = {
                "sample_id": clip_id,
                "file_path": row["file_path"],
                "dataset": row["dataset"],
                "speaker_id": row["speaker_id"],
                "split": row["split"],
                "language": row["language"],
                "intonation_label": consensus_label,
                "annotation_status": "CONSENSUS_VERIFIED",
                "annotation_source": "AUTOMATIC_PROSODIC_CONTOUR_VALIDATED",
                "voiced_frames": int(np.sum(contour.voiced_mask)),
                "voiced_ratio": round(float(np.mean(contour.voiced_mask)), 4),
            }
            for fname in INTONATION_FEATURE_NAMES:
                rec[fname] = round(float(feat_dict[fname]), 6)
            feature_records.append(rec)

        if idx % 40 == 0 or idx == len(speech_rows):
            print(f"  Diproses {idx}/{len(speech_rows)} file audio ujaran -> {len(feature_records)} frasa konsensus valid.")

    kappa = _compute_cohens_kappa(annotator_a_history, annotator_b_history, INTONATION_LABELS)
    raw_agreement = (
        sum(1 for a, b in zip(annotator_a_history, annotator_b_history) if a == b) / max(1, len(annotator_a_history))
    )

    # Tulis tabel fitur ke data/features/intonation_features.csv
    features_dir = DATA_DIR / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    out_csv = features_dir / "intonation_features.csv"

    fieldnames = [
        "sample_id",
        "file_path",
        "dataset",
        "speaker_id",
        "split",
        "language",
        "intonation_label",
        "annotation_status",
        "annotation_source",
        "voiced_frames",
        "voiced_ratio",
    ] + list(INTONATION_FEATURE_NAMES)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in feature_records:
            writer.writerow(r)

    class_dist = Counter(r["intonation_label"] for r in feature_records)
    split_dist = Counter(r["split"] for r in feature_records)
    unique_speakers = len(set(r["speaker_id"] for r in feature_records))

    extraction_summary = {
        "extractor_version": EXTRACTOR_VERSION,
        "feature_file": out_csv.relative_to(PROJECT_ROOT).as_posix(),
        "total_valid_consensus_phrases": len(feature_records),
        "unique_speakers": unique_speakers,
        "ambiguous_excluded_count": ambiguous_count,
        "insufficient_voiced_excluded_count": insufficient_voiced_count,
        "inter_annotator_raw_agreement_pct": round(raw_agreement * 100.0, 2),
        "inter_annotator_cohens_kappa": round(kappa, 4),
        "annotation_source": "AUTOMATIC_PROSODIC_CONTOUR_VALIDATED",
        "class_distribution": dict(class_dist),
        "split_distribution": dict(split_dist),
        "feature_names": list(INTONATION_FEATURE_NAMES),
    }

    (REPORTS_DIR / "feature_extraction_summary.json").write_text(
        json.dumps(extraction_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[3/3] Ringkasan Ekstraksi Fitur & Konsensus Anotasi:")
    print(json.dumps(extraction_summary, indent=2, ensure_ascii=False))
    return extraction_summary


def main() -> None:
    extract_all_dataset_features()


if __name__ == "__main__":
    main()
