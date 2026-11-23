"""Halaman 5: Evaluasi Model (Informasi Sekunder untuk Developer & Penelitian Pengenalan Pola)."""

import json
import pandas as pd
import streamlit as st

from src.config import EVALUATION_REPORT_PATH
from src.prediction import check_intonation_model_status
from src.visualization import create_confusion_matrix_figure

st.set_page_config(page_title="Evaluasi Model — Voice Analyzer", page_icon="🧪", layout="wide")

st.title("🧪 Evaluasi Model SVM & Hasil Eksperimen Pengenalan Pola")
st.write(
    "Halaman ini menyajikan laporan evaluasi nyata dari model klasifikasi pola intonasi **Support Vector Machine (SVM)** "
    "serta evaluasi akurasi ekstraksi $F_0$ terhadap referensi laryngograph **PTDB-TUG**."
)

model_status = check_intonation_model_status()

if not EVALUATION_REPORT_PATH.exists():
    st.warning(model_status.message)
    st.info(
        "Laporan evaluasi belum tersedia di `reports/evaluation_report.json`. "
        "Jalankan pipeline ekstraksi dan pelatihan offline:\n"
        "```powershell\n"
        "python -m training.extract_dataset_features\n"
        "python -m training.train_intonation\n"
        "python -m training.evaluate\n"
        "```"
    )
else:
    report = json.loads(EVALUATION_REPORT_PATH.read_text(encoding="utf-8"))
    m_test = report.get("metrics_test", {})
    m_dev = report.get("metrics_train_val", {})
    split_sum = report.get("split_summary", {})

    if model_status.is_ready:
        st.success(
            f"✅ **Model Aktif:** `{report.get('model_version')}` | "
            f"**Pipeline:** `{report.get('pipeline_description')}` | "
            f"**Tanggal Training:** `{report.get('trained_at_utc')}`"
        )
    else:
        st.warning(model_status.message)

    # Baris Metrik Utama pada Speaker-Independent Test Set
    st.markdown("### 📊 1. Performa pada Speaker-Independent Test Set")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Macro-F1 (Test Set)", f"{m_test.get('macro_f1', 0):.4f}")
    k2.metric("Balanced Accuracy", f"{m_test.get('balanced_accuracy', 0):.4f}")
    k3.metric("Accuracy", f"{m_test.get('accuracy', 0):.4f}")
    k4.metric("CV Macro-F1 (GroupKFold)", f"{m_dev.get('cv_macro_f1', 0):.4f}")
    k5.metric(
        "Sampel & Pembicara Uji",
        f"{split_sum.get('test_samples', 0)} frasa",
        f"{split_sum.get('test_speakers', 0)} pembicara baru (0% leakage)",
        delta_color="off",
    )

    # Tabel Per Kelas & Confusion Matrix
    col_cm, col_cls = st.columns([1, 1])
    with col_cm:
        st.markdown("### 🟦 2. Visualisasi Confusion Matrix (Test Set)")
        cm_data = report.get("confusion_matrix", {})
        if cm_data.get("matrix") and cm_data.get("labels"):
            fig_cm = create_confusion_matrix_figure(
                cm=cm_data["matrix"],
                labels=cm_data["labels"],
                title="Confusion Matrix — Speaker-Independent Test Set",
            )
            st.plotly_chart(fig_cm)

    with col_cls:
        st.markdown("### 📋 3. Precision, Recall, dan F1-Score per Kelas")
        per_cls = report.get("per_class_metrics", [])
        if per_cls:
            df_cls = pd.DataFrame(per_cls).rename(
                columns={
                    "class_label": "Kelas Intonasi",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1_score": "F1-Score",
                    "support_test": "Sampel Test",
                    "support_train_val": "Sampel Train/Val",
                }
            )
            st.dataframe(df_cls, width="stretch", hide_index=True)

        st.markdown("#### 🔍 Analisis Overfitting / Underfitting & Misclassification")
        st.info(report.get("overfitting_underfitting_analysis", ""))
        top_mis = report.get("top_misclassifications", [])
        if top_mis:
            st.markdown("**Pasangan Kelas yang Paling Sering Tertukar (*Misclassification*):**")
            for item in top_mis:
                st.markdown(
                    f"- Kelas sebenarnya **`{item['true_label']}`** diprediksi sebagai **`{item['predicted_label']}`**: "
                    f"**{item['count']} sampel**"
                )

    st.divider()

    # Skenario Eksperimen (Membandingkan Baseline, Kernel SVM, Hyperparameter, dan PCA)
    st.markdown("### 🧪 4. Perbandingan Skenario Eksperimen (Baseline vs Variasi Kernel SVM & PCA)")
    scenarios = report.get("experiment_scenarios", [])
    if scenarios:
        df_scen = pd.DataFrame(scenarios).rename(
            columns={
                "scenario_name": "Skenario Eksperimen",
                "description": "Konfigurasi Pipeline",
                "hyperparameters": "Hyperparameter Terbaik",
                "train_macro_f1": "Train Macro-F1",
                "cv_macro_f1": "CV Macro-F1",
                "test_macro_f1": "Test Macro-F1",
                "test_balanced_accuracy": "Test Balanced Acc",
                "test_accuracy": "Test Accuracy",
                "train_test_f1_gap": "Gap (Train - Test)",
                "fit_diagnosis": "Diagnosis Fit",
            }
        )
        st.dataframe(df_scen, width="stretch", hide_index=True)

    st.divider()

    # Evaluasi Referensi F0 pada PTDB-TUG & Metadata Pembagian Split
    col_f0, col_meta = st.columns(2)
    with col_f0:
        st.markdown("### 🎚️ 5. Evaluasi Pengukuran F0 terhadap Referensi PTDB-TUG")
        f0_eval = report.get("f0_reference_evaluation")
        if f0_eval:
            st.write(f"- **Dataset Referensi:** `{f0_eval.get('reference_dataset')}`")
            st.write(f"- **Jumlah Klip Dievaluasi:** `{f0_eval.get('evaluated_clips')}` klip")
            st.write(f"- **Median Fine Pitch Error (FPE):** `{f0_eval.get('median_fine_pitch_error_cents')} cents`")
            st.write(f"- **Mean Gross Pitch Error (GPE > 20%):** `{f0_eval.get('mean_gross_pitch_error_pct')}%`")
            st.write(f"- **Mean Voicing Decision Error (VDE):** `{f0_eval.get('mean_voicing_decision_error_pct')}%`")

    with col_meta:
        st.markdown("### 🗂️ 6. Rincian Split, Anotasi & Parameter Model")
        ann_q = report.get("annotation_quality") or {}
        st.write(f"- **Metode Split:** `{split_sum.get('split_method')}`")
        st.write(
            f"- **Distribusi Split:** Train/Val = `{split_sum.get('train_val_samples')} sampel` "
            f"(`{split_sum.get('train_val_speakers')} pembicara`), "
            f"Test = `{split_sum.get('test_samples')} sampel` (`{split_sum.get('test_speakers')} pembicara`)"
        )
        st.write(f"- **Sumber Anotasi Kontur:** `{ann_q.get('annotation_source', 'AUTOMATIC_PROSODIC_CONTOUR_VALIDATED')}`")
        st.write(
            f"- **Kesepakatan Antar-Penilai Kontur:** Raw Agreement = `{ann_q.get('inter_annotator_raw_agreement_pct')}%`, "
            f"Cohen's Kappa = `{ann_q.get('inter_annotator_cohens_kappa')}` "
            f"(Kontur ambigu disisihkan: `{ann_q.get('ambiguous_excluded_count')} sampel`)"
        )
        st.write(f"- **Hyperparameter SVM Terpilih:** `{json.dumps(report.get('best_hyperparameters'))}`")
