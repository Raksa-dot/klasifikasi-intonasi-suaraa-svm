"""Ekspor ringkasan hasil latihan dan perbandingan percobaan ke format JSON dan CSV."""

import json
from typing import Any, Dict, List
import pandas as pd


def export_summary_to_json(data: Dict[str, Any]) -> str:
    """Konversi dictionary hasil analisis ke string JSON terformat."""
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def export_summary_to_csv(data: Dict[str, Any]) -> str:
    """Konversi ringkasan metrik datar ke string CSV."""
    flat = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}
    df = pd.DataFrame([flat])
    return df.to_csv(index=False)


def export_attempts_list_to_csv(attempts: List[Dict[str, Any]]) -> str:
    """Konversi daftar percobaan sesi ke tabel CSV ringkas untuk diunduh pengguna."""
    rows: List[Dict[str, Any]] = []
    for att in attempts:
        row: Dict[str, Any] = {
            "attempt_id": att.get("attempt_id"),
            "timestamp": att.get("timestamp"),
            "mode": att.get("mode"),
            "exercise_id": att.get("exercise_id"),
            "target": att.get("target"),
            "phrase_text": att.get("phrase_text", ""),
            "predicted_label": att.get("predicted_label", ""),
        }
        summary = att.get("summary") or {}
        for k, v in summary.items():
            if not isinstance(v, (list, dict)):
                row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)
