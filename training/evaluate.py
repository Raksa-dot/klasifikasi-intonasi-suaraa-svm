"""Evaluasi menyeluruh model SVM intonasi pada speaker-independent test set dan pembuatan laporan di reports/."""

from collections import Counter
import csv
import json
from typing import Any, Dict, List
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import (
    EVALUATION_REPORT_PATH,
    EXPERIMENT_SCENARIOS_PATH,
    INTONATION_FEATURE_NAMES,
    INTONATION_LABELS,
    INTONATION_METADATA_PATH,
    INTONATION_MODEL_PATH,
    REPORTS_DIR,
)
from training.train_intonation import load_feature_dataset


def evaluate_trained_model() -> Dict[str, Any]:
    """Evaluasi model SVM pada test set terpisah dan simpan laporan lengkap ke reports/evaluation_report.json."""
    if not INTONATION_MODEL_PATH.exists() or not INTONATION_METADATA_PATH.exists():
        raise FileNotFoundError("Artefak model belum ditemukan. Jalankan `python -m training.train_intonation` terlebih dahulu.")

    pipeline = joblib.load(INTONATION_MODEL_PATH)
    metadata = json.loads(INTONATION_METADATA_PATH.read_text(encoding="utf-8"))

    rows, X, y, speakers, splits = load_feature_dataset()
    train_val_mask = np.isin(splits, ["train", "val"])
    test_mask = splits == "test"

    X_dev, y_dev, spk_dev = X[train_val_mask], y[train_val_mask], speakers[train_val_mask]
    X_test, y_test, spk_test = X[test_mask], y[test_mask], speakers[test_mask]

    y_pred_dev = pipeline.predict(X_dev)
    y_pred_test = pipeline.predict(X_test)

    labels = [lbl for lbl in INTONATION_LABELS if lbl in set(y)]

    # Metrik pada Test Set (Speaker-Independent)
    test_acc = float(accuracy_score(y_test, y_pred_test))
    test_bal_acc = float(balanced_accuracy_score(y_test, y_pred_test))
    test_macro_f1 = float(f1_score(y_test, y_pred_test, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0))

    # Metrik pada Train/Val Set untuk analisis Overfitting / Underfitting
    dev_acc = float(accuracy_score(y_dev, y_pred_dev))
    dev_macro_f1 = float(f1_score(y_dev, y_pred_dev, average="macro", zero_division=0))

    prec, rec, f1_cls, support = precision_recall_fscore_support(
        y_test, y_pred_test, labels=labels, zero_division=0
    )

    per_class_metrics: List[Dict[str, Any]] = []
    for idx, lbl in enumerate(labels):
        per_class_metrics.append(
            {
                "class_label": lbl,
                "precision": round(float(prec[idx]), 4),
                "recall": round(float(rec[idx]), 4),
                "f1_score": round(float(f1_cls[idx]), 4),
                "support_test": int(support[idx]),
                "support_train_val": int(np.sum(y_dev == lbl)),
            }
        )

    cm = confusion_matrix(y_test, y_pred_test, labels=labels).tolist()

    # Analisis pasangan kesalahan klasifikasi (misclassification) terbanyak
    misclassifications: List[Dict[str, Any]] = []
    for i, true_lbl in enumerate(labels):
        for j, pred_lbl in enumerate(labels):
            if i != j and cm[i][j] > 0:
                misclassifications.append(
                    {
                        "true_label": true_lbl,
                        "predicted_label": pred_lbl,
                        "count": int(cm[i][j]),
                    }
                )
    misclassifications.sort(key=lambda x: -x["count"])

    # Baca laporan tambahan jika tersedia
    f0_ref_path = REPORTS_DIR / "f0_reference_evaluation.json"
    f0_ref_report = json.loads(f0_ref_path.read_text(encoding="utf-8")) if f0_ref_path.exists() else None

    feat_sum_path = REPORTS_DIR / "feature_extraction_summary.json"
    feat_summary = json.loads(feat_sum_path.read_text(encoding="utf-8")) if feat_sum_path.exists() else None

    scenarios_list: List[Dict[str, Any]] = []
    if EXPERIMENT_SCENARIOS_PATH.exists():
        with EXPERIMENT_SCENARIOS_PATH.open("r", encoding="utf-8") as f:
            scenarios_list = list(csv.DictReader(f))

    f1_gap = dev_macro_f1 - test_macro_f1
    if test_macro_f1 >= 0.70 and abs(f1_gap) <= 0.18:
        overfitting_analysis = (
            f"Model menunjukkan generalisasi yang baik pada pembicara baru (Good Fit): "
            f"Train/Val Macro-F1 = {dev_macro_f1:.4f}, CV Macro-F1 = {metadata.get('cv_macro_f1', 0):.4f}, "
            f"dan Speaker-Independent Test Macro-F1 = {test_macro_f1:.4f} (selisih {f1_gap:+.4f}). "
            "Tidak ditemukan indikasi underfitting maupun overfitting berat."
        )
    elif f1_gap > 0.18:
        overfitting_analysis = (
            f"Terdapat selisih performa antara Train/Val Macro-F1 ({dev_macro_f1:.4f}) dan Test Macro-F1 ({test_macro_f1:.4f}) "
            f"sebesar {f1_gap:+.4f} yang menunjukkan variasi prosodi antar-pembicara baru pada himpunan uji."
        )
    else:
        overfitting_analysis = (
            f"Train/Val Macro-F1 = {dev_macro_f1:.4f}, Test Macro-F1 = {test_macro_f1:.4f} (selisih {f1_gap:+.4f})."
        )

    report: Dict[str, Any] = {
        "model_version": metadata.get("model_version"),
        "trained_at_utc": metadata.get("trained_at_utc"),
        "pipeline_description": metadata.get("pipeline_description"),
        "best_hyperparameters": metadata.get("best_hyperparameters"),
        "extractor_version": metadata.get("extractor_version"),
        "feature_names": list(INTONATION_FEATURE_NAMES),
        "labels": labels,
        "split_summary": {
            "split_method": metadata.get("dataset_summary", {}).get("split_method"),
            "zero_speaker_leakage": len(set(spk_dev).intersection(set(spk_test))) == 0,
            "train_val_samples": int(len(y_dev)),
            "train_val_speakers": int(len(set(spk_dev))),
            "test_samples": int(len(y_test)),
            "test_speakers": int(len(set(spk_test))),
            "total_samples": int(len(y)),
            "total_speakers": int(len(set(speakers))),
            "samples_per_split": dict(Counter(splits)),
        },
        "metrics_test": {
            "macro_f1": round(test_macro_f1, 4),
            "weighted_f1": round(test_weighted_f1, 4),
            "balanced_accuracy": round(test_bal_acc, 4),
            "accuracy": round(test_acc, 4),
        },
        "metrics_train_val": {
            "train_val_macro_f1": round(dev_macro_f1, 4),
            "cv_macro_f1": metadata.get("cv_macro_f1"),
            "train_val_accuracy": round(dev_acc, 4),
            "train_test_macro_f1_gap": round(f1_gap, 4),
        },
        "overfitting_underfitting_analysis": overfitting_analysis,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm,
        },
        "top_misclassifications": misclassifications[:5],
        "experiment_scenarios": scenarios_list,
        "f0_reference_evaluation": f0_ref_report,
        "annotation_quality": feat_summary,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ LAPORAN EVALUASI MODEL SVM (SPEAKER-INDEPENDENT TEST SET) ================")
    print(f"Model Version      : {report['model_version']}")
    print(f"Pipeline           : {report['pipeline_description']}")
    print(f"Best Hyperparams   : {report['best_hyperparameters']}")
    print(f"Test Macro-F1      : {test_macro_f1:.4f}")
    print(f"Test Balanced Acc  : {test_bal_acc:.4f}")
    print(f"Test Accuracy      : {test_acc:.4f}")
    print(f"Analisis Fit       : {overfitting_analysis}")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, labels=labels, digits=4, zero_division=0))
    return report


def main() -> None:
    evaluate_trained_model()


if __name__ == "__main__":
    main()
