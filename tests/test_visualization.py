"""Unit test untuk modul visualisasi grafik Plotly (src/visualization.py)."""

import numpy as np
import plotly.graph_objects as go
import pytest

from src.config import INTONATION_LABELS
from src.visualization import (
    create_comparison_overlay_figure,
    create_confusion_matrix_figure,
    create_pitch_contour_figure,
    create_target_guide_figure,
    create_waveform_figure,
)


def test_create_target_guide_figure_all_classes():
    """Pastikan seluruh pola intonasi dapat dirender tanpa error properti Plotly."""
    for label in INTONATION_LABELS:
        fig = create_target_guide_figure(label)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.layout.template is not None
        assert fig.layout.plot_bgcolor == "#090d16"
        assert fig.layout.xaxis.title.text == "Progres Frasa (%)"
        assert fig.layout.yaxis.title.text == "Relatif (Semitone)"

    # Fallback unknown label
    fig_unk = create_target_guide_figure("UNKNOWN")
    assert isinstance(fig_unk, go.Figure)


def test_create_waveform_figure():
    """Pastikan waveform audio PCM dapat dirender dengan benar."""
    samples = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1.0, 22050))
    fig = create_waveform_figure(samples, sample_rate=22050, time_offset_sec=0.1)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.layout.xaxis.title.text == "Waktu (detik)"
    assert fig.layout.yaxis.title.text == "Amplitudo Normal"


def test_create_pitch_contour_figure_with_target_and_tolerance():
    """Pastikan kontur pitch dengan garis target dan zona toleransi cents tidak crash."""
    t = np.linspace(0, 2.0, 100)
    f0 = np.full(100, 440.0)
    f0[10:20] = np.nan  # Simulasi unvoiced gap

    fig = create_pitch_contour_figure(
        timestamps=t,
        f0_values=f0,
        y_label="Frekuensi Dasar F0 (Hz)",
        title="Uji Kontur F0",
        target_hz=440.0,
        target_label="Target A4 (440.0 Hz)",
        tolerance_cents=50.0,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    # Check shapes (target line + tolerance rect)
    assert len(fig.layout.shapes) >= 2


def test_create_comparison_overlay_figure():
    """Pastikan visualisasi overlay komparasi dua percobaan bekerja normal."""
    ta = np.linspace(0, 1.5, 60)
    va = np.linspace(-1.0, 2.0, 60)
    tb = np.linspace(0, 2.0, 80)
    vb = np.linspace(-0.8, 1.8, 80)

    # Test tanpa normalisasi waktu
    fig1 = create_comparison_overlay_figure(
        time_a=ta,
        val_a=va,
        label_a="Percobaan 1",
        time_b=tb,
        val_b=vb,
        label_b="Percobaan 2",
        normalize_time=False,
        y_label="Frekuensi (Hz)",
        target_line_y=220.0,
        target_line_label="Target A3",
    )
    assert isinstance(fig1, go.Figure)
    assert len(fig1.data) == 2
    assert fig1.layout.xaxis.title.text == "Waktu (detik)"

    # Test dengan normalisasi waktu (0 - 100%)
    fig2 = create_comparison_overlay_figure(
        time_a=ta,
        val_a=va,
        label_a="Percobaan 1",
        time_b=tb,
        val_b=vb,
        label_b="Percobaan 2",
        normalize_time=True,
        y_label="Semitone Relatif",
    )
    assert isinstance(fig2, go.Figure)
    assert len(fig2.data) == 2
    assert fig2.layout.xaxis.title.text == "Durasi Ternormalisasi (%)"


def test_create_confusion_matrix_figure():
    """Pastikan visualisasi confusion matrix heat map bekerja normal."""
    cm = [
        [10, 1, 0, 0, 0],
        [0, 12, 0, 1, 0],
        [1, 0, 15, 0, 0],
        [0, 0, 0, 8, 2],
        [0, 0, 0, 1, 14],
    ]
    labels = list(INTONATION_LABELS)
    fig = create_confusion_matrix_figure(cm, labels, title="Evaluasi Confusion Matrix")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.layout.xaxis.title.text == "Kelas Prediksi Model"
    assert fig.layout.yaxis.title.text == "Kelas Sebenarnya (Ground Truth)"
