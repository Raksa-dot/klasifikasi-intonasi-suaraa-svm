"""Visualisasi grafik waveform, kontur pitch, ilustrasi panduan target, dan perbandingan percobaan."""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def create_target_guide_figure(target_label: str) -> go.Figure:
    """Buat grafik ilustrasi kontur panduan untuk latihan berbicara (diberi label tegas sebagai panduan)."""
    t = np.linspace(0.0, 100.0, 100)
    if target_label == "RISING":
        y = -1.5 + 3.5 * (t / 100.0) ** 1.4
    elif target_label == "FALLING":
        y = 1.5 - 3.5 * (t / 100.0) ** 1.3
    elif target_label == "LEVEL":
        y = 0.15 * np.sin(2.0 * np.pi * t / 100.0)
    elif target_label == "FALL_RISE":
        x = t / 100.0
        y = 6.5 * (x - 0.45) ** 2 - 1.4
    elif target_label == "RISE_FALL":
        x = t / 100.0
        y = -6.5 * (x - 0.55) ** 2 + 1.4
    else:
        y = np.zeros_like(t)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t.tolist(),
            y=y.tolist(),
            mode="lines",
            name=f"Panduan {target_label}",
            line=dict(color="#2563eb", width=3, dash="dot"),
        )
    )
    fig.update_layout(
        title=f"Ilustrasi Panduan Pola Target: {target_label} (Panduan Visual, Bukan Hasil Model)",
        xaxis_title="Persentase Durasi Frasa (%)",
        yaxis_title="Perubahan Pitch Relatif (Semitone Ilustratif)",
        template="plotly_white",
        height=260,
        margin=dict(l=40, r=20, t=45, b=35),
    )
    return fig


def create_waveform_figure(samples: np.ndarray, sample_rate: int, time_offset_sec: float = 0.0) -> go.Figure:
    """Buat grafik waveform sinyal audio terhadap waktu."""
    n = samples.size
    step = max(1, n // 2500)
    y = samples[::step]
    t = (np.arange(0, n, step) / max(1, sample_rate)) + time_offset_sec

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t.tolist(),
            y=y.tolist(),
            mode="lines",
            name="Waveform",
            line=dict(color="#475569", width=1.2),
        )
    )
    fig.update_layout(
        title="Waveform Sinyal Audio",
        xaxis_title="Waktu (detik)",
        yaxis_title="Amplitudo",
        template="plotly_white",
        height=240,
        margin=dict(l=40, r=20, t=40, b=35),
    )
    return fig


def create_pitch_contour_figure(
    timestamps: np.ndarray,
    f0_values: np.ndarray,
    y_label: str = "Pitch / F0 (Hz)",
    title: str = "Kontur Pitch terhadap Waktu",
    target_hz: Optional[float] = None,
    target_label: Optional[str] = None,
    tolerance_cents: Optional[float] = None,
) -> go.Figure:
    """Buat grafik kontur pitch dengan mempertahankan bagian non-voiced sebagai celah (NaN)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.asarray(timestamps).tolist(),
            y=np.asarray(f0_values).tolist(),
            mode="lines+markers",
            name="Pitch Valid (Voiced)",
            connectgaps=False,
            line=dict(color="#0f766e", width=2.5),
            marker=dict(size=5),
        )
    )
    if target_hz is not None and target_hz > 0:
        label = target_label or f"Nada Target ({target_hz:.2f} Hz)"
        fig.add_hline(
            y=target_hz,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=label,
            annotation_position="top left",
        )
        if tolerance_cents is not None and tolerance_cents > 0:
            upper_hz = target_hz * (2.0 ** (tolerance_cents / 1200.0))
            lower_hz = target_hz * (2.0 ** (-tolerance_cents / 1200.0))
            fig.add_hrect(
                y0=lower_hz,
                y1=upper_hz,
                fillcolor="#16a34a",
                opacity=0.12,
                line_width=0,
                annotation_text=f"Zona Toleransi (±{tolerance_cents:.0f} cents)",
                annotation_position="bottom right",
            )

    fig.update_layout(
        title=title,
        xaxis_title="Waktu (detik)",
        yaxis_title=y_label,
        template="plotly_white",
        height=380,
        margin=dict(l=45, r=25, t=50, b=40),
    )
    return fig


def create_comparison_overlay_figure(
    time_a: np.ndarray,
    val_a: np.ndarray,
    label_a: str,
    time_b: np.ndarray,
    val_b: np.ndarray,
    label_b: str,
    normalize_time: bool = False,
    y_label: str = "Pitch Relatif (Semitone)",
    title: str = "Perbandingan Kontur Pitch Antarpercobaan",
    target_line_y: Optional[float] = None,
    target_line_label: Optional[str] = None,
) -> go.Figure:
    """Buat grafik overlay dua percobaan pada mode dan target latihan yang sama."""
    fig = go.Figure()

    def _prep_x(t_arr: np.ndarray) -> np.ndarray:
        if not normalize_time or t_arr.size < 2:
            return t_arr
        t_min = float(np.nanmin(t_arr))
        t_max = float(np.nanmax(t_arr))
        span = max(1e-6, t_max - t_min)
        return ((t_arr - t_min) / span) * 100.0

    xa = _prep_x(np.asarray(time_a, dtype=float))
    xb = _prep_x(np.asarray(time_b, dtype=float))

    fig.add_trace(
        go.Scatter(
            x=xa.tolist(),
            y=np.asarray(val_a).tolist(),
            mode="lines+markers",
            name=label_a,
            connectgaps=False,
            line=dict(color="#2563eb", width=2.5),
            marker=dict(size=5, symbol="circle"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xb.tolist(),
            y=np.asarray(val_b).tolist(),
            mode="lines+markers",
            name=label_b,
            connectgaps=False,
            line=dict(color="#d97706", width=2.5, dash="dash"),
            marker=dict(size=6, symbol="diamond"),
        )
    )

    if target_line_y is not None:
        fig.add_hline(
            y=target_line_y,
            line_dash="dot",
            line_color="#dc2626",
            annotation_text=target_line_label or "Target",
        )

    x_title = "Persentase Durasi Rekaman (%)" if normalize_time else "Waktu Asli (detik)"
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_label,
        template="plotly_white",
        height=400,
        margin=dict(l=45, r=25, t=50, b=40),
    )
    return fig


def create_confusion_matrix_figure(cm: List[List[int]], labels: List[str], title: str = "Confusion Matrix") -> go.Figure:
    """Buat visualisasi heatmap Confusion Matrix menggunakan Plotly."""
    z = np.asarray(cm, dtype=int)
    text_vals = [[str(v) for v in row] for row in z]
    fig = go.Figure(
        data=go.Heatmap(
            z=z.tolist(),
            x=list(labels),
            y=list(labels),
            text=text_vals,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Kelas Prediksi Model (Predicted Label)",
        yaxis_title="Kelas Sebenarnya (True Label)",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=440,
        margin=dict(l=60, r=30, t=55, b=50),
    )
    return fig
