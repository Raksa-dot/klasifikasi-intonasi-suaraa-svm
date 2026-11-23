"""Konfigurasi terpusat untuk pengolahan audio, batas kualitas, fitur F0, dan artefak model."""

from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

EXTRACTOR_VERSION = "f0-pyin-v1.0"

# Konfigurasi input audio
INTERNAL_SAMPLE_RATE = 22050
MAX_UPLOAD_SIZE_MB = 10
MIN_DURATION_SEC = 0.6
MAX_DURATION_SEC = 15.0

# Ambang pemeriksaan kualitas sinyal (diukur dari sinyal asli sebelum normalisasi)
MIN_RMS_ENERGY = 0.003
MAX_CLIPPING_RATIO = 0.05
CLIPPING_THRESHOLD = 0.99
MIN_VOICED_FRAMES = 10
MIN_VOICED_RATIO = 0.15

# Parameter ekstraksi F0 (librosa.pyin)
FRAME_LENGTH = 2048
HOP_LENGTH = 256
SPEECH_FMIN_HZ = 65.0
SPEECH_FMAX_HZ = 550.0
SINGING_FMIN_HZ = 65.0
SINGING_FMAX_HZ = 1000.0

# Tuning musik dan parameter latihan bernyanyi
A4_REFERENCE_HZ = 440.0
A4_MIDI_NUMBER = 69
DEFAULT_CENTS_TOLERANCE = 50.0

# Kelas intonasi untuk model SVM
INTONATION_LABELS: Tuple[str, ...] = (
    "RISING",
    "FALLING",
    "LEVEL",
    "FALL_RISE",
    "RISE_FALL",
)

LABEL_DESCRIPTIONS_ID = {
    "RISING": "Intonasi Naik (pitch meningkat menuju akhir frasa)",
    "FALLING": "Intonasi Turun (pitch menurun menuju akhir frasa)",
    "LEVEL": "Intonasi Datar (pitch relatif stabil sepanjang frasa)",
    "FALL_RISE": "Turun lalu Naik (pitch turun di awal/tengah lalu naik di akhir)",
    "RISE_FALL": "Naik lalu Turun (pitch naik ke puncak di tengah lalu turun di akhir)",
}

# Daftar nama fitur intonasi (urutan wajib konsisten untuk training, evaluasi, dan inferensi website)
INTONATION_FEATURE_NAMES: Tuple[str, ...] = (
    "pitch_range_semitones",
    "std_semitones",
    "iqr_semitones",
    "p90_minus_p10_semitones",
    "overall_slope_st_per_sec",
    "overall_slope_st_normalized",
    "start_mean_st",
    "mid_mean_st",
    "end_mean_st",
    "end_minus_start_st",
    "mid_minus_start_st",
    "end_minus_mid_st",
    "poly2_curvature",
    "peak_minus_valley_pos",
)

INTONATION_MODEL_PATH = MODELS_DIR / "intonation_svm.joblib"
INTONATION_METADATA_PATH = MODELS_DIR / "intonation_svm_metadata.json"
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.json"
EXPERIMENT_SCENARIOS_PATH = REPORTS_DIR / "experiment_scenarios.csv"

MODEL_UNAVAILABLE_MESSAGE = (
    "Klasifikasi intonasi belum tersedia; grafik dan pengukuran pitch tetap dapat digunakan untuk latihan."
)
