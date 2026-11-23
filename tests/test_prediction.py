"""Unit test untuk pengecekan status jujur dan inferensi model SVM."""

from src.config import INTONATION_FEATURE_NAMES, INTONATION_LABELS
from src.prediction import check_intonation_model_status, predict_intonation_pattern


def test_model_status_and_prediction_contract() -> None:
    status = check_intonation_model_status()
    dummy_features = {name: 0.0 for name in INTONATION_FEATURE_NAMES}
    res = predict_intonation_pattern(dummy_features)
    if status.is_ready:
        assert res.is_available
        assert res.predicted_label in INTONATION_LABELS
    else:
        assert not res.is_available
        assert res.predicted_label is None
        assert "Klasifikasi intonasi belum tersedia" in res.status_message
