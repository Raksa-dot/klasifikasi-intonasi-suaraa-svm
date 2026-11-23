"""Pemuatan dan inferensi model SVM dengan verifikasi metadata dan status jujur."""

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional
import joblib
import numpy as np

from src.config import (
    EXTRACTOR_VERSION,
    INTONATION_FEATURE_NAMES,
    INTONATION_METADATA_PATH,
    INTONATION_MODEL_PATH,
    MODEL_UNAVAILABLE_MESSAGE,
)


@dataclass
class ModelStatus:
    """Status ketersediaan dan kecocokan metadata model SVM."""

    is_ready: bool
    message: str
    model_version: Optional[str] = None
    supported_labels: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PredictionResult:
    """Hasil inferensi model SVM pada satu rekaman frasa pendek."""

    is_available: bool
    predicted_label: Optional[str]
    status_message: str
    model_version: Optional[str] = None
    decision_scores: Optional[Dict[str, float]] = None


def check_intonation_model_status() -> ModelStatus:
    """Periksa apakah artefak model intonasi dan metadatanya tersedia serta cocok."""
    if not INTONATION_MODEL_PATH.exists() or not INTONATION_METADATA_PATH.exists():
        return ModelStatus(is_ready=False, message=MODEL_UNAVAILABLE_MESSAGE)

    try:
        with INTONATION_METADATA_PATH.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return ModelStatus(is_ready=False, message=MODEL_UNAVAILABLE_MESSAGE)

    if metadata.get("extractor_version") != EXTRACTOR_VERSION:
        return ModelStatus(
            is_ready=False,
            message="Versi feature extractor model tidak cocok dengan aplikasi saat ini. " + MODEL_UNAVAILABLE_MESSAGE,
        )

    expected_features = list(INTONATION_FEATURE_NAMES)
    if metadata.get("feature_names") != expected_features:
        return ModelStatus(
            is_ready=False,
            message="Urutan/daftar fitur pada metadata model tidak cocok. " + MODEL_UNAVAILABLE_MESSAGE,
        )

    return ModelStatus(
        is_ready=True,
        message="Model klasifikasi intonasi SVM siap digunakan.",
        model_version=metadata.get("model_version"),
        supported_labels=metadata.get("labels"),
        metadata=metadata,
    )


def predict_intonation_pattern(feature_dict: Optional[Dict[str, float]]) -> PredictionResult:
    """Jalankan prediksi pola intonasi menggunakan artefak SVM yang tervalidasi."""
    if feature_dict is None:
        return PredictionResult(
            is_available=False,
            predicted_label=None,
            status_message="Data pitch valid belum mencukupi untuk klasifikasi intonasi.",
        )

    status = check_intonation_model_status()
    if not status.is_ready:
        return PredictionResult(
            is_available=False,
            predicted_label=None,
            status_message=status.message,
        )

    try:
        pipeline = joblib.load(INTONATION_MODEL_PATH)
        x_vec = np.array([[float(feature_dict[name]) for name in INTONATION_FEATURE_NAMES]], dtype=float)
        if not np.all(np.isfinite(x_vec)):
            return PredictionResult(
                is_available=False,
                predicted_label=None,
                status_message="Vektor fitur mengandung nilai tidak valid (NaN/Inf).",
            )

        pred_label = str(pipeline.predict(x_vec)[0])
        decision_scores: Optional[Dict[str, float]] = None
        if hasattr(pipeline, "decision_function"):
            raw_scores = pipeline.decision_function(x_vec)
            classes = list(getattr(pipeline, "classes_", []))
            if raw_scores.ndim == 2 and len(classes) == raw_scores.shape[1]:
                decision_scores = {str(cls): float(raw_scores[0, idx]) for idx, cls in enumerate(classes)}

        return PredictionResult(
            is_available=True,
            predicted_label=pred_label,
            status_message=f"Pola intonasi diprediksi oleh model SVM ({status.model_version}).",
            model_version=status.model_version,
            decision_scores=decision_scores,
        )
    except Exception as exc:
        return PredictionResult(
            is_available=False,
            predicted_label=None,
            status_message=f"Gagal memuat atau menjalankan model SVM: {exc}",
        )
