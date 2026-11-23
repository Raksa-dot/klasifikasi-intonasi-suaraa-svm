"""Visualisasi grafik waveform, kontur pitch, ilustrasi panduan target, dan perbandingan percobaan dengan tema modern."""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _apply_clean_theme(
    fig: go.Figure,
    title: str,
    height: int = 360,
    xaxis_title: str = "",
    yaxis_title: str = "",
) -> go.Figure:
    """Terapkan tema grafik minimalis, bersih, dan profesional (kontras tinggi, grid halus)."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13, color="#0f172a", family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
            x=0.01,
            xanchor="left",
        ),
        xaxis=dict(
            title=xaxis_title,
            showgrid=True,
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            tickfont=dict(size=11, color="#64748b"),
            titlefont=dict(size=12, color="#334155"),
        ),
        yaxis=dict(
            title=yaxis_title,
            showgrid=True,
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            tickfont=dict(size=11, color="#64748b"),
            titlefont=dict(size=12, color="#334155"),
        ),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=height,
        margin=dict(l=45, r=25, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#475569"),
        ),
    )
    return fig


def create_target_guide_figure(target_label: str) -> go.Figure:
    """Buat grafik ilustrasi kontur panduan untuk latihan berbicara."""
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
            line=dict(color="#0284c7", width=2.5, dash="dot"),
        )
    )
    return _apply_clean_theme(
        fig,
        title=f"Panduan Pola Target: {target_label} (Referensi Visual)",
        height=240,
        xaxis_title="Progres Frasa (%)",
        yaxis_title="Relatif (Semitone)",
    )


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
            name="Amplitudo",
            line=dict(color="#64748b", width=1.0),
        )
    )
    return _apply_clean_theme(
        fig,
        title="Waveform Sinyal Audio (PCM Domain)",
        height=200,
        xaxis_title="Waktu (detik)",
        yaxis_title="Amplitudo Normal",
    )


def create_pitch_contour_figure(
    timestamps: np.ndarray,
    f0_values: np.ndarray,
    y_label: str = "Frekuensi Dasar / F0 (Hz)",
    title: str = "Kontur Pitch Sinyal Suara",
    target_hz: Optional[float] = None,
    target_label: Optional[str] = None,
    tolerance_cents: Optional[float] = None,
) -> go.Figure:
    """Buat grafik kontur pitch dengan garis halus dan penanganan celah unvoiced."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.asarray(timestamps).tolist(),
            y=np.asarray(f0_values).tolist(),
            mode="lines+markers",
            name="Pitch Terdeteksi (Voiced)",
            connectgaps=False,
            line=dict(color="#0284c7", width=2.2),
            marker=dict(size=4, color="#0284c7"),
        )
    )
    if target_hz is not None and target_hz > 0:
        label = target_label or f"Target: {target_hz:.1f} Hz"
        fig.add_hline(
            y=target_hz,
            line_dash="dash",
            line_color="#dc2626",
            line_width=1.5,
            annotation_text=label,
            annotation_position="top left",
            annotation_font=dict(size=11, color="#dc2626"),
        )
        if tolerance_cents is not None and tolerance_cents > 0:
            upper_hz = target_hz * (2.0 ** (tolerance_cents / 1200.0))
            lower_hz = target_hz * (2.0 ** (-tolerance_cents / 1200.0))
            fig.add_hrect(
                y0=lower_hz,
                y1=upper_hz,
                fillcolor="#059669",
                opacity=0.08,
                line_width=0,
                annotation_text=f"Toleransi ±{tolerance_cents:.0f} c",
                annotation_position="bottom right",
                annotation_font=dict(size=10, color="#059669"),
            )

    return _apply_clean_theme(
        fig,
        title=title,
        height=360,
        xaxis_title="Waktu (detik)",
        yaxis_title=y_label,
    )


def create_comparison_overlay_figure(
    time_a: np.ndarray,
    val_a: np.ndarray,
    label_a: str,
    time_b: np.ndarray,
    val_b: np.ndarray,
    label_b: str,
    normalize_time: bool = False,
    y_label: str = "Pitch Relatif (Semitone)",
    title: str = "Perbandingan Kontur Pitch",
    target_line_y: Optional[float] = None,
    target_line_label: Optional[str] = None,
) -> go.Figure:
    """Buat grafik overlay dua percobaan latihan."""
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
            line=dict(color="#0284c7", width=2.2),
            marker=dict(size=4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xb.tolist(),
            y=np.asarray(val_b).tolist(),
            mode="lines+markers",
            name=label_b,
            connectgaps=False,
            line=dict(color="#d97706", width=2.2, dash="dash"),
            marker=dict(size=4),
        )
    )

    if target_line_y is not None:
        fig.add_hline(
            y=target_line_y,
            line_dash="dot",
            line_color="#dc2626",
            annotation_text=target_line_label or "Target",
            annotation_font=dict(size=11, color="#dc2626"),
        )

    x_title = "Durasi Ternormalisasi (%)" if normalize_time else "Waktu (detik)"
    return _apply_clean_theme(
        fig,
        title=title,
        height=380,
        xaxis_title=x_title,
        yaxis_title=y_label,
    )


def create_confusion_matrix_figure(cm: List[List[int]], labels: List[str], title: str = "Matriks Kebingungan (Confusion Matrix)") -> go.Figure:
    """Visualisasi Confusion Matrix modern dengan palet warna Blues kontras tinggi."""
    z = np.asarray(cm, dtype=int)
    text_vals = [[str(v) for v in row] for row in z]
    fig = go.Figure(
        data=go.Heatmap(
            z=z.tolist(),
            x=list(labels),
            y=list(labels),
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#0f172a"),
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13, color="#0f172a", family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
            x=0.01,
            xanchor="left",
        ),
        xaxis=dict(
            title="Kelas Prediksi Model",
            tickfont=dict(size=11, color="#334155"),
            titlefont=dict(size=12, color="#0f172a"),
        ),
        yaxis=dict(
            title="Kelas Sebenarnya (Ground Truth)",
            autorange="reversed",
            tickfont=dict(size=11, color="#334155"),
            titlefont=dict(size=12, color="#0f172a"),
        ),
        template="plotly_white",
        height=400,
        margin=dict(l=65, r=25, t=50, b=50),
    )
    return fig
