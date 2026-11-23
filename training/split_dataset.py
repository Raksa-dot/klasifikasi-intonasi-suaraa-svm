"""Pembagian dataset berdasarkan speaker_id (Speaker-Independent Split) dan verifikasi anti-leakage."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Dict, List

from src.config import DATA_DIR, REPORTS_DIR


def verify_and_report_splits(metadata_csv_path: Path | None = None) -> Dict[str, object]:
    """Periksa dan laporkan distribusi split berbasis pembicara pada metadata.csv."""
    if metadata_csv_path is None:
        metadata_csv_path = DATA_DIR / "annotations" / "metadata.csv"

    with metadata_csv_path.open("r", encoding="utf-8") as f:
        rows: List[Dict[str, str]] = list(csv.DictReader(f))

    # Verifikasi 0% speaker leakage lintas split
    speaker_splits: Dict[str, set] = defaultdict(set)
    for r in rows:
        speaker_splits[r["speaker_id"]].add(r["split"])

    leaked_speakers = [spk for spk, s_set in speaker_splits.items() if len(s_set) > 1]
    if leaked_speakers:
        raise RuntimeError(f"Terdeteksi kebocoran speaker lintas split: {leaked_speakers}")

    by_split = Counter(r["split"] for r in rows)
    by_dataset = Counter(r["dataset"] for r in rows)

    speakers_per_split: Dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        speakers_per_split[split_name] = len(
            set(r["speaker_id"] for r in rows if r["split"] == split_name)
        )

    summary = {
        "total_samples": len(rows),
        "total_unique_speakers": len(speaker_splits),
        "zero_speaker_leakage_verified": True,
        "samples_per_split": dict(by_split),
        "speakers_per_split": speakers_per_split,
        "samples_per_dataset": dict(by_dataset),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "split_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    summary = verify_and_report_splits()
    print("[Speaker-Independent Split Verified]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
