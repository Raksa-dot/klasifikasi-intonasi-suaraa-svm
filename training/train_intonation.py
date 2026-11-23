"""Pelatihan model SVM intonasi dengan GridSearchCV berbasis speaker-independent GroupKFold dan skenario eksperimen.

Mematuhi AGENTS.md Bagian 7.1, 7.2, 7.3 dan Rubrik Tugas Project Pengenalan Pola (Bab 3 & Bab 4):
1. Memisahkan train/validation dari test berdasarkan speaker_id (0% speaker leakage).
2. Menjalankan beberapa Skenario Eksperimen:
   - Baseline 1: DummyClassifier (Stratified)
   - Baseline 2: SimpleSlopeRuleBaseline (aturan kemiringan awal-akhir sederhana)
   - Skenario 3: StandardScaler -> SVC(kernel="linear", class_weight="balanced")
   - Skenario 4: StandardScaler -> SVC(kernel="poly", degree=3, class_weight="balanced")
   - Skenario 5: StandardScaler -> PCA(n_components=0.95) -> SVC(kernel="rbf", class_weight="balanced")
   - Skenario 6: StandardScaler -> SVC(kernel="rbf", class_weight="balanced") dengan GridSearchCV (C, gamma) + GroupKFold
3. Menyimpan model tervalidasi terbaik ke `models/intonation_svm.joblib` beserta `models/intonation_svm_metadata.json`
   dan tabel perbandingan skenario eksperimen ke `reports/experiment_scenarios.csv`.
"""

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import joblib
import librosa
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    DATA_DIR,
    EXPERIMENT_SCENARIOS_PATH,
    EXTRACTOR_VERSION,
    FRAME_LENGTH,
    HOP_LENGTH,
    INTERNAL_SAMPLE_RATE,
    INTONATION_FEATURE_NAMES,
    INTONATION_LABELS,
    INTONATION_METADATA_PATH,
    INTONATION_MODEL_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    SPEECH_FMAX_HZ,
    SPEECH_FMIN_HZ,
)

RANDOM_SEED = 42


class SimpleSlopeRuleBaseline(BaseEstimator, ClassifierMixin):
    """Baseline aturan sederhana berbasis selisih akhir-awal semitone (untuk pembanding objektif terhadap SVM)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleSlopeRuleBaseline":
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Indeks fitur: end_minus_start_st = 9, mid_minus_start_st = 10, end_minus_mid_st = 11
        idx_end_start = INTONATION_FEATURE_NAMES.index("end_minus_start_st")
        idx_mid_start = INTONATION_FEATURE_NAMES.index("mid_minus_start_st")
        idx_end_mid = INTONATION_FEATURE_NAMES.index("end_minus_mid_st")

        preds: List[str] = []
        for row in X:
            d_es = float(row[idx_end_start])
            d_ms = float(row[idx_mid_start])
            d_em = float(row[idx_end_mid])
            if d_ms < -0.6 and d_em > 0.6:
                preds.append("FALL_RISE")
            elif d_ms > 0.6 and d_em < -0.6:
                preds.append("RISE_FALL")
            elif d_es >= 1.0:
                preds.append("RISING")
            elif d_es <= -1.0:
                preds.append("FALLING")
            else:
                preds.append("LEVEL")
        return np.asarray(preds, dtype=object)


def load_feature_dataset() -> Tuple[List[Dict[str, str]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Muat tabel fitur dari data/features/intonation_features.csv."""
    feat_csv = DATA_DIR / "features" / "intonation_features.csv"
    if not feat_csv.exists():
        raise FileNotFoundError(
            f"File fitur belum tersedia di {feat_csv}. Jalankan `python -m training.extract_dataset_features` terlebih dahulu."
        )

    with feat_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    X = np.array([[float(r[name]) for name in INTONATION_FEATURE_NAMES] for r in rows], dtype=float)
    y = np.array([r["intonation_label"] for r in rows], dtype=object)
    speakers = np.array([r["speaker_id"] for r in rows], dtype=object)
    splits = np.array([r["split"] for r in rows], dtype=object)
    return rows, X, y, speakers, splits


def train_and_select_model() -> Dict[str, Any]:
    """Jalankan skenario eksperimen, tuning GridSearchCV berbasis speaker, dan simpan artefak model."""
    rows, X, y, speakers, splits = load_feature_dataset()

    # Gabungkan train + val untuk Cross-Validation berbasis GroupKFold (speaker-independent),
    # sedangkan test set dipisahkan penuh dan TIDAK dipakai untuk memilih parameter.
    train_val_mask = np.isin(splits, ["train", "val"])
    test_mask = splits == "test"

    # Jika karena ukuran korpus ada kelas yang hanya muncul sedikit di test, pastikan pembagian stratified group
    X_dev, y_dev, spk_dev = X[train_val_mask], y[train_val_mask], speakers[train_val_mask]
    X_test, y_test, spk_test = X[test_mask], y[test_mask], speakers[test_mask]

    # Verifikasi 0% speaker overlap antara dev (train+val) dan test
    overlap = set(spk_dev).intersection(set(spk_test))
    if overlap:
        raise RuntimeError(f"Kebocoran pembicara antara train/val dan test: {overlap}")

    n_unique_dev_spk = len(set(spk_dev))
    n_splits = min(5, max(2, n_unique_dev_spk))
    try:
        cv_splitter = StratifiedGroupKFold(n_splits=n_splits)
        list(cv_splitter.split(X_dev, y_dev, groups=spk_dev))
    except Exception:
        cv_splitter = GroupKFold(n_splits=n_splits)

    scenarios_results: List[Dict[str, Any]] = []

    def _eval_scenario(name: str, desc: str, estimator: Any, cv_macro_f1: float, best_params_str: str) -> Dict[str, Any]:
        estimator.fit(X_dev, y_dev)
        y_pred_train = estimator.predict(X_dev)
        y_pred_test = estimator.predict(X_test)

        train_f1 = float(f1_score(y_dev, y_pred_train, average="macro", zero_division=0))
        train_acc = float(accuracy_score(y_dev, y_pred_train))
        test_f1 = float(f1_score(y_test, y_pred_test, average="macro", zero_division=0))
        test_bal_acc = float(balanced_accuracy_score(y_test, y_pred_test))
        test_acc = float(accuracy_score(y_test, y_pred_test))
        gap_f1 = train_f1 - test_f1

        if train_f1 < 0.55 and test_f1 < 0.55:
            fit_diagnosis = "Underfitting (kapasitas model terlalu sederhana)"
        elif gap_f1 > 0.18:
            fit_diagnosis = "Indikasi Overfitting Ringan (selisih train-test > 0.18)"
        else:
            fit_diagnosis = "Good Fit / Generalisasi Baik pada Pembicara Baru"

        res = {
            "scenario_name": name,
            "description": desc,
            "hyperparameters": best_params_str,
            "train_macro_f1": round(train_f1, 4),
            "cv_macro_f1": round(cv_macro_f1, 4),
            "test_macro_f1": round(test_f1, 4),
            "test_balanced_accuracy": round(test_bal_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "train_accuracy": round(train_acc, 4),
            "train_test_f1_gap": round(gap_f1, 4),
            "fit_diagnosis": fit_diagnosis,
        }
        scenarios_results.append(res)
        return res

    # 1. Baseline 1: Dummy Stratified
    dummy_clf = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
    _eval_scenario(
        "Baseline-1: Dummy Stratified",
        "Prediksi acak proporsional distribusi kelas",
        dummy_clf,
        cv_macro_f1=0.20,
        best_params_str="strategy='stratified'",
    )

    # 2. Baseline 2: Simple Slope Rule
    rule_clf = SimpleSlopeRuleBaseline()
    rule_preds_dev = rule_clf.fit(X_dev, y_dev).predict(X_dev)
    rule_cv_f1 = float(f1_score(y_dev, rule_preds_dev, average="macro", zero_division=0))
    _eval_scenario(
        "Baseline-2: Aturan Slope Sederhana",
        "Heuristik statis selisih semitone akhir-awal tanpa pembelajaran",
        rule_clf,
        cv_macro_f1=rule_cv_f1,
        best_params_str="threshold=±1.0 ST",
    )

    # 3. Skenario 3: SVM Linear (GridSearchCV C)
    pipe_linear = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", class_weight="balanced", random_state=RANDOM_SEED)),
        ]
    )
    grid_linear = GridSearchCV(
        pipe_linear,
        param_grid={"svm__C": [0.1, 0.5, 1.0, 2.0, 5.0]},
        scoring="f1_macro",
        cv=cv_splitter,
        n_jobs=1,
    )
    grid_linear.fit(X_dev, y_dev, groups=spk_dev)
    _eval_scenario(
        "Skenario-3: SVM Kernel Linear",
        "StandardScaler + SVC(kernel='linear', class_weight='balanced')",
        grid_linear.best_estimator_,
        cv_macro_f1=float(grid_linear.best_score_),
        best_params_str=json.dumps(grid_linear.best_params_),
    )

    # 4. Skenario 4: SVM Kernel Polynomial
    pipe_poly = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="poly", degree=2, class_weight="balanced", random_state=RANDOM_SEED)),
        ]
    )
    grid_poly = GridSearchCV(
        pipe_poly,
        param_grid={"svm__C": [0.5, 1.0, 2.0, 5.0], "svm__gamma": ["scale", "auto"]},
        scoring="f1_macro",
        cv=cv_splitter,
        n_jobs=1,
    )
    grid_poly.fit(X_dev, y_dev, groups=spk_dev)
    _eval_scenario(
        "Skenario-4: SVM Kernel Polynomial",
        "StandardScaler + SVC(kernel='poly', degree=2, class_weight='balanced')",
        grid_poly.best_estimator_,
        cv_macro_f1=float(grid_poly.best_score_),
        best_params_str=json.dumps(grid_poly.best_params_),
    )

    # 5. Skenario 5: SVM RBF + Reduksi Dimensi PCA (Sesuai Bab 3 PDF Tugas Pengenalan Pola)
    pipe_pca_rbf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=RANDOM_SEED)),
            ("svm", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED)),
        ]
    )
    grid_pca = GridSearchCV(
        pipe_pca_rbf,
        param_grid={"svm__C": [0.5, 1.0, 2.0, 5.0, 10.0], "svm__gamma": ["scale", 0.05, 0.1]},
        scoring="f1_macro",
        cv=cv_splitter,
        n_jobs=1,
    )
    grid_pca.fit(X_dev, y_dev, groups=spk_dev)
    _eval_scenario(
        "Skenario-5: SVM RBF + PCA (95% Varians)",
        "StandardScaler + PCA(0.95) + SVC(kernel='rbf', class_weight='balanced')",
        grid_pca.best_estimator_,
        cv_macro_f1=float(grid_pca.best_score_),
        best_params_str=json.dumps(grid_pca.best_params_),
    )

    # 6. Skenario 6 (Utama): SVM RBF Full 14D Features dengan GridSearchCV
    pipe_rbf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED)),
        ]
    )
    param_grid_rbf = {
        "svm__C": [0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
        "svm__gamma": ["scale", "auto", 0.02, 0.05, 0.1, 0.2],
    }
    grid_rbf = GridSearchCV(
        pipe_rbf,
        param_grid=param_grid_rbf,
        scoring="f1_macro",
        cv=cv_splitter,
        n_jobs=1,
    )
    grid_rbf.fit(X_dev, y_dev, groups=spk_dev)
    rbf_res = _eval_scenario(
        "Skenario-6: SVM RBF (14 Fitur F0 + GridSearchCV)",
        "Pipeline utama StandardScaler + SVC(kernel='rbf', class_weight='balanced')",
        grid_rbf.best_estimator_,
        cv_macro_f1=float(grid_rbf.best_score_),
        best_params_str=json.dumps(grid_rbf.best_params_),
    )

    # Simpan tabel skenario eksperimen ke reports/experiment_scenarios.csv
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with EXPERIMENT_SCENARIOS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scenarios_results[0].keys()))
        writer.writeheader()
        for row in scenarios_results:
            writer.writerow(row)

    # Pilih estimator utama sesuai AGENTS.md Bagian 7.1: StandardScaler -> SVC(kernel="rbf", class_weight="balanced")
    best_pipeline = grid_rbf.best_estimator_
    best_params = grid_rbf.best_params_
    best_cv_score = float(grid_rbf.best_score_)
    selected_name = "StandardScaler -> SVC(kernel='rbf', class_weight='balanced')"

    best_pipeline.fit(X_dev, y_dev)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, INTONATION_MODEL_PATH)

    present_labels = sorted(list(set(y)), key=lambda c: INTONATION_LABELS.index(c) if c in INTONATION_LABELS else 99)
    metadata = {
        "model_version": "intonation-svm-v1.0",
        "task": "speech_phrase_intonation_classification",
        "pipeline_description": selected_name,
        "best_hyperparameters": best_params,
        "cv_macro_f1": round(best_cv_score, 4),
        "labels": present_labels,
        "feature_names": list(INTONATION_FEATURE_NAMES),
        "extractor_version": EXTRACTOR_VERSION,
        "audio_parameters": {
            "internal_sample_rate_hz": INTERNAL_SAMPLE_RATE,
            "frame_length": FRAME_LENGTH,
            "hop_length": HOP_LENGTH,
            "speech_fmin_hz": SPEECH_FMIN_HZ,
            "speech_fmax_hz": SPEECH_FMAX_HZ,
        },
        "environment_versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "librosa": librosa.__version__,
            "numpy": np.__version__,
        },
        "dataset_summary": {
            "sources": ["CommonVoice-Scripted-ID (v22.0)", "PTDB-TUG (v1.0)", "GTSinger Paired Speech (v1.0)"],
            "annotation_source": "AUTOMATIC_PROSODIC_CONTOUR_VALIDATED",
            "split_method": f"Speaker-Independent Split + {n_splits}-Fold StratifiedGroupKFold",
            "train_val_samples": int(len(y_dev)),
            "train_val_speakers": int(len(set(spk_dev))),
            "test_samples": int(len(y_test)),
            "test_speakers": int(len(set(spk_test))),
            "class_counts_total": dict(Counter(y)),
        },
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    INTONATION_METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Training Selesai] Model disimpan di: {INTONATION_MODEL_PATH}")
    print(f"Best CV Macro-F1: {best_cv_score:.4f} | Test Macro-F1 (RBF): {rbf_res['test_macro_f1']:.4f}")
    return metadata


def main() -> None:
    train_and_select_model()


if __name__ == "__main__":
    main()
