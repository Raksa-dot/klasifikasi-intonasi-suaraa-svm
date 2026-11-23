"""Komponen UI bersama Streamlit: Desain Minimalis Profesional, Input Audio, dan Live Instrument Tuner."""

import hashlib
from typing import Any, Dict, Optional, Tuple
import streamlit as st
import streamlit.components.v1 as components

from src.config import DATA_DIR, MAX_DURATION_SEC, MAX_UPLOAD_SIZE_MB, MIN_DURATION_SEC


def inject_custom_css() -> None:
    """Suntikkan styling antarmuka modern Dark Mode profesional (palet slate gelap & cyan presisi)."""
    st.markdown(
        """
        <style>
        /* Tipografi & Smoothing Dark Mode */
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
            background-color: #090d16 !important;
            color: #f1f5f9 !important;
        }

        /* Hierarki Judul Elegan Dark */
        h1 {
            font-weight: 700 !important;
            letter-spacing: -0.03em !important;
            color: #f8fafc !important;
            margin-bottom: 0.35rem !important;
            font-size: 2.1rem !important;
        }
        h2 {
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
            color: #f1f5f9 !important;
            margin-top: 1.6rem !important;
            margin-bottom: 0.6rem !important;
            font-size: 1.35rem !important;
        }
        h3 {
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            color: #e2e8f0 !important;
            font-size: 1.1rem !important;
        }
        p, label, .stMarkdown p {
            color: #cbd5e1;
        }

        /* Container Kartu Dark Slate */
        .va-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 14px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            transition: border-color 0.15s ease;
        }
        .va-card:hover {
            border-color: #334155;
        }

        .va-header-tag {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #38bdf8 !important;
            background: rgba(56, 189, 248, 0.12) !important;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
            border-radius: 4px;
            padding: 2px 8px;
            margin-bottom: 8px;
        }

        /* Metric Box Streamlit Dark */
        [data-testid="stMetric"] {
            background: #0f172a !important;
            border: 1px solid #1e293b !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
            box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.2) !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            color: #94a3b8 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.45rem !important;
            font-weight: 700 !important;
            color: #f8fafc !important;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        }

        /* Tombol & Aksi Dark Mode */
        div.stButton > button {
            border-radius: 6px !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            padding: 0.4rem 1rem !important;
            border: 1px solid #334155 !important;
            background: #0f172a !important;
            color: #f1f5f9 !important;
            transition: all 0.15s ease-in-out !important;
        }
        div.stButton > button:hover {
            border-color: #64748b !important;
            background: #1e293b !important;
            color: #ffffff !important;
        }
        div.stButton > button[kind="primary"] {
            background: #0284c7 !important;
            border-color: #38bdf8 !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #0369a1 !important;
            border-color: #0284c7 !important;
        }

        /* Navigasi Tab Dark */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: 1px solid #1e293b !important;
            background-color: transparent !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            font-size: 0.88rem;
            font-weight: 500;
            color: #94a3b8 !important;
            background-color: transparent !important;
        }
        .stTabs [aria-selected="true"] {
            color: #38bdf8 !important;
            border-bottom-color: #38bdf8 !important;
            font-weight: 600 !important;
        }

        /* Expander & Divider Dark */
        div[data-testid="stExpander"] {
            border: 1px solid #1e293b !important;
            background-color: #0f172a !important;
            border-radius: 8px !important;
        }
        hr {
            border-color: #1e293b !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    """Inisialisasi struktur state sesi latihan secara terisolasi per sesi browser."""
    if "attempts" not in st.session_state:
        st.session_state["attempts"] = []
    if "current_analysis" not in st.session_state:
        st.session_state["current_analysis"] = None
    if "active_context_signature" not in st.session_state:
        st.session_state["active_context_signature"] = ""
    if "input_reset_counter" not in st.session_state:
        st.session_state["input_reset_counter"] = 0


def reset_entire_session() -> None:
    """Bersihkan seluruh input audio, hasil analisis aktif, dan kedua/seluruh percobaan di sesi ini."""
    st.session_state["attempts"] = []
    st.session_state["current_analysis"] = None
    st.session_state["active_context_signature"] = ""
    st.session_state["input_reset_counter"] = st.session_state.get("input_reset_counter", 0) + 1


def clear_current_input_and_result() -> None:
    """Bersihkan hasil analisis saat ini dan ganti key widget input untuk merekam ulang."""
    st.session_state["current_analysis"] = None
    st.session_state["active_context_signature"] = ""
    st.session_state["input_reset_counter"] = st.session_state.get("input_reset_counter", 0) + 1


def render_live_pitch_monitor(
    mode: str = "SPEAKING",
    target_hz: Optional[float] = None,
    target_label: str = "",
) -> None:
    """Tampilkan oscilloscope pitch real-time bergaya instrumen audio studio."""
    target_hz_js = f"{target_hz:.4f}" if (target_hz is not None and target_hz > 0) else "null"
    mode_title = (
        f"Kontur Intonasi Waktu-Nyata (Target: {target_label})"
        if mode == "SPEAKING"
        else f"Tuner Nada Waktu-Nyata (Target: {target_label} — {target_hz:.2f} Hz)"
        if target_hz
        else "Monitor Pitch Waktu-Nyata"
    )

    html_code = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; border: 1px solid #1e293b; border-radius: 8px; padding: 14px 18px; background: #090d16; color: #f8fafc;">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; border-bottom: 1px solid #1e293b; padding-bottom: 10px;">
        <div>
          <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #38bdf8; font-weight: 600;">Oscilloscope Pitch Web Audio</div>
          <div style="color: #f1f5f9; font-size: 14px; font-weight: 600;">{mode_title}</div>
        </div>
        <div>
          <button id="btnStartLive" style="background: #0284c7; color: white; border: none; border-radius: 5px; padding: 6px 14px; font-weight: 500; cursor: pointer; font-size: 12px; letter-spacing: 0.02em;">Mulai Pemantauan</button>
          <button id="btnStopLive" style="background: #ef4444; color: white; border: none; border-radius: 5px; padding: 6px 14px; font-weight: 500; cursor: pointer; font-size: 12px; display: none;">Hentikan</button>
        </div>
      </div>

      <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 12px;">
        <div style="background: #0f172a; padding: 8px 12px; border-radius: 6px; border: 1px solid #1e293b;">
          <div style="font-size: 10px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Frekuensi Dasar F0</div>
          <div id="liveHz" style="font-size: 19px; font-weight: 700; color: #38bdf8; font-family: monospace;">— Hz</div>
        </div>
        <div style="background: #0f172a; padding: 8px 12px; border-radius: 6px; border: 1px solid #1e293b;">
          <div style="font-size: 10px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Notasi Nada</div>
          <div id="liveNote" style="font-size: 19px; font-weight: 700; color: #34d399; font-family: monospace;">—</div>
        </div>
        <div style="background: #0f172a; padding: 8px 12px; border-radius: 6px; border: 1px solid #1e293b;">
          <div style="font-size: 10px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Deviasi Cents</div>
          <div id="liveCents" style="font-size: 19px; font-weight: 700; color: #fbbf24; font-family: monospace;">—</div>
        </div>
        <div style="background: #0f172a; padding: 8px 12px; border-radius: 6px; border: 1px solid #1e293b;">
          <div style="font-size: 10px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Status Vokal</div>
          <div id="livePattern" style="font-size: 13px; font-weight: 600; color: #cbd5e1; padding-top: 4px;">Menunggu input...</div>
        </div>
      </div>

      <canvas id="pitchCanvas" width="760" height="130" style="width: 100%; height: 130px; background: #020617; border: 1px solid #1e293b; border-radius: 6px; display: block;"></canvas>
      <div id="liveStatus" style="font-size: 11px; color: #94a3b8; margin-top: 8px;">
        Monitor ini memproses sinyal mikrofon langsung di browser untuk panduan vokal instan. Gunakan modul input di bawah untuk analisis matematis penuh (pyin + SVM).
      </div>
    </div>

    <script>
    (function() {{
      const NOTE_STRINGS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
      const targetHz = {target_hz_js};
      let audioCtx = null;
      let analyser = null;
      let mediaStream = null;
      let rafId = null;
      const historyHz = [];
      const MAX_POINTS = 140;

      const btnStart = document.getElementById("btnStartLive");
      const btnStop = document.getElementById("btnStopLive");
      const elHz = document.getElementById("liveHz");
      const elNote = document.getElementById("liveNote");
      const elCents = document.getElementById("liveCents");
      const elPattern = document.getElementById("livePattern");
      const elStatus = document.getElementById("liveStatus");
      const canvas = document.getElementById("pitchCanvas");
      const ctx = canvas.getContext("2d");

      function autoCorrelate(buf, sampleRate) {{
        let size = buf.length;
        let rms = 0;
        for (let i = 0; i < size; i++) {{
          const val = buf[i];
          rms += val * val;
        }}
        rms = Math.sqrt(rms / size);
        if (rms < 0.012) return -1;

        let r1 = 0, r2 = size - 1, thres = 0.2;
        for (let i = 0; i < size / 2; i++) {{
          if (Math.abs(buf[i]) < thres) {{ r1 = i; break; }}
        }}
        for (let i = 1; i < size / 2; i++) {{
          if (Math.abs(buf[size - i]) < thres) {{ r2 = size - i; break; }}
        }}

        buf = buf.slice(r1, r2);
        size = buf.length;
        const c = new Float32Array(size);
        for (let i = 0; i < size; i++) {{
          for (let j = 0; j < size - i; j++) {{
            c[i] = c[i] + buf[j] * buf[j + i];
          }}
        }}

        let d = 0;
        while (c[d] > c[d + 1]) d++;
        let maxval = -1, maxpos = -1;
        for (let i = d; i < size; i++) {{
          if (c[i] > maxval) {{
            maxval = c[i];
            maxpos = i;
          }}
        }}
        let T0 = maxpos;
        if (T0 <= 0) return -1;
        return sampleRate / T0;
      }}

      function drawCanvas() {{
        ctx.fillStyle = "#020617";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Grid lines
        ctx.strokeStyle = "#0f172a";
        ctx.lineWidth = 1;
        for (let y = 20; y < canvas.height; y += 25) {{
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(canvas.width, y);
          ctx.stroke();
        }}

        const minF = 70.0;
        const maxF = 550.0;

        if (targetHz && targetHz >= minF && targetHz <= maxF) {{
          const ty = canvas.height - ((targetHz - minF) / (maxF - minF)) * (canvas.height - 20) - 10;
          ctx.strokeStyle = "#ef4444";
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(0, ty);
          ctx.lineTo(canvas.width, ty);
          ctx.stroke();
          ctx.setLineDash([]);
        }}

        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < historyHz.length; i++) {{
          const f = historyHz[i];
          const x = (i / (MAX_POINTS - 1)) * canvas.width;
          if (f === null || f <= 0) {{
            started = false;
            continue;
          }}
          const y = canvas.height - ((f - minF) / (maxF - minF)) * (canvas.height - 20) - 10;
          if (!started) {{
            ctx.moveTo(x, y);
            started = true;
          }} else {{
            ctx.lineTo(x, y);
          }}
        }}
        ctx.stroke();
      }}

      function updatePitch() {{
        if (!analyser) return;
        const buf = new Float32Array(2048);
        analyser.getFloatTimeDomainData(buf);
        const f0 = autoCorrelate(buf, audioCtx.sampleRate);

        if (f0 !== -1 && f0 >= 60 && f0 <= 800) {{
          historyHz.push(f0);
          const midi = 69 + 12 * Math.log2(f0 / 440.0);
          const nearestMidi = Math.round(midi);
          const noteName = NOTE_STRINGS[(nearestMidi % 12 + 12) % 12] + (Math.floor(nearestMidi / 12) - 1);
          const refFreq = targetHz ? targetHz : 440.0 * Math.pow(2, (nearestMidi - 69) / 12.0);
          const cents = 1200 * Math.log2(f0 / refFreq);

          elHz.textContent = f0.toFixed(1) + " Hz";
          elNote.textContent = noteName;
          elCents.textContent = (cents >= 0 ? "+" : "") + cents.toFixed(1) + " c";
          elPattern.textContent = "Sinyal Terdeteksi";
        }} else {{
          historyHz.push(null);
        }}

        if (historyHz.length > MAX_POINTS) historyHz.shift();
        drawCanvas();
        rafId = requestAnimationFrame(updatePitch);
      }}

      btnStart.addEventListener("click", async () => {{
        try {{
          mediaStream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          const source = audioCtx.createMediaStreamSource(mediaStream);
          analyser = audioCtx.createAnalyser();
          analyser.fftSize = 2048;
          source.connect(analyser);
          btnStart.style.display = "none";
          btnStop.style.display = "inline-block";
          elStatus.textContent = "Status: Pemantauan mikrofon aktif. Bersuara untuk melihat kontur.";
          updatePitch();
        }} catch (err) {{
          elStatus.textContent = "Peringatan: Izin mikrofon browser tidak aktif (" + err.message + ").";
        }}
      }});

      btnStop.addEventListener("click", () => {{
        if (rafId) cancelAnimationFrame(rafId);
        if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
        if (audioCtx) audioCtx.close();
        analyser = null;
        btnStop.style.display = "none";
        btnStart.style.display = "inline-block";
        elStatus.textContent = "Status: Pemantauan dihentikan.";
      }});
    }})();
    </script>
    """
    if hasattr(st, "iframe"):
        st.iframe(html_code, height=295)
    else:
        components.html(html_code, height=295)


def render_shared_audio_input(
    mode: str,
    exercise_id: str,
    target_key: str,
) -> Tuple[Optional[bytes], str, bool]:
    """Tampilkan komponen input audio bersama dengan layout profesional."""
    init_session_state()
    reset_cnt = st.session_state.get("input_reset_counter", 0)

    st.markdown("### Masukan Sinyal Audio")
    st.caption(
        f"Spesifikasi: Format PCM WAV/MP3, durasi rekomendasi {MIN_DURATION_SEC:.1f} – {MAX_DURATION_SEC:.1f} detik, ukuran maks. {MAX_UPLOAD_SIZE_MB} MB."
    )

    tab_mic, tab_upload, tab_demo = st.tabs([
        "Perekam Mikrofon",
        "Unggah Berkas Audio",
        "Sampel Rekaman Uji",
    ])

    audio_bytes: Optional[bytes] = None
    filename_hint = "recording.wav"
    source_label = ""

    mic_val = None
    with tab_mic:
        st.write("Tekan ikon mikrofon untuk merekam satu frasa atau nada, lalu hentikan rekaman:")
        mic_audio = st.audio_input(
            "Perekam Suara Mikrofon:",
            key=f"mic_{mode}_{reset_cnt}",
            help="Perekam langsung menggunakan Web Audio API browser.",
        )
        if mic_audio is not None:
            mic_val = mic_audio.getvalue()
            if mic_val != st.session_state.get(f"last_seen_mic_{mode}"):
                st.session_state[f"last_seen_mic_{mode}"] = mic_val
                st.session_state[f"active_input_type_{mode}"] = "mic"
            st.success("Sinyal rekaman mikrofon berhasil ditangkap.")

    up_val = None
    up_name = "uploaded.wav"
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Pilih berkas audio (WAV, MP3, FLAC, OGG):",
            type=["wav", "mp3", "flac", "ogg"],
            key=f"upload_{mode}_{reset_cnt}",
        )
        if uploaded_file is not None:
            up_val = uploaded_file.getvalue()
            up_name = uploaded_file.name or "uploaded.wav"
            if up_val != st.session_state.get(f"last_seen_upload_{mode}"):
                st.session_state[f"last_seen_upload_{mode}"] = up_val
                st.session_state[f"active_input_type_{mode}"] = "upload"
            st.success(f"Berkas audio {up_name} ({len(up_val)/1024:.1f} KB) siap diproses.")

    with tab_demo:
        st.write("Pilih sampel rekaman terverifikasi dari dataset untuk pengujian langsung:")
        demo_files = []
        if mode == "SPEAKING":
            sp_dir = DATA_DIR / "raw" / "speech" / "common_voice_id"
            if sp_dir.exists():
                for p in sorted(sp_dir.glob("*.mp3"))[:5]:
                    demo_files.append((p.name, p))
        else:
            sg_dir = DATA_DIR / "raw" / "singing" / "gtsinger"
            if sg_dir.exists():
                for p in sorted(sg_dir.glob("*.wav"))[:5]:
                    demo_files.append((p.name, p))

        if demo_files:
            demo_idx = st.selectbox(
                "Daftar sampel dataset:",
                range(len(demo_files)),
                format_func=lambda i: f"Sampel {i+1}: {demo_files[i][0]}",
                key=f"demo_sel_{mode}_{reset_cnt}",
            )
            sel_path = demo_files[demo_idx][1]
            if st.button("Gunakan Sampel Ini Sebagai Input", key=f"btn_demo_{mode}_{reset_cnt}"):
                st.session_state[f"demo_audio_bytes_{mode}"] = sel_path.read_bytes()
                st.session_state[f"demo_audio_name_{mode}"] = demo_files[demo_idx][0]
                st.session_state[f"active_input_type_{mode}"] = "demo"
                st.rerun()

        demo_stored = st.session_state.get(f"demo_audio_bytes_{mode}")
        demo_name = st.session_state.get(f"demo_audio_name_{mode}", "demo_audio.wav")
        if demo_stored:
            st.success(f"Sampel uji {demo_name} aktif sebagai masukan.")

    # Tentukan sumber input audio aktif secara deterministik
    active_type = st.session_state.get(f"active_input_type_{mode}")
    if active_type == "demo" and demo_stored:
        audio_bytes = demo_stored
        filename_hint = demo_name
        source_label = f"Sampel Uji ({demo_name})"
    elif active_type == "upload" and up_val:
        audio_bytes = up_val
        filename_hint = up_name
        source_label = f"Berkas Unggahan ({up_name})"
    elif active_type == "mic" and mic_val:
        audio_bytes = mic_val
        filename_hint = getattr(mic_audio, "name", "mic_recording.wav") or "mic_recording.wav"
        source_label = "Perekam Mikrofon"
    else:
        # Fallback fleksibel ke input yang tersedia
        if mic_val is not None:
            audio_bytes = mic_val
            filename_hint = "mic_recording.wav"
            source_label = "Perekam Mikrofon"
        elif up_val is not None:
            audio_bytes = up_val
            filename_hint = up_name
            source_label = f"Berkas Unggahan ({up_name})"
        elif demo_stored is not None:
            audio_bytes = demo_stored
            filename_hint = demo_name
            source_label = f"Sampel Uji ({demo_name})"

    # Sinkronisasi state
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()[:16] if audio_bytes else "no_audio"
    context_sig = f"{mode}|{exercise_id}|{target_key}|{audio_hash}"

    if st.session_state.get("active_context_signature") != context_sig:
        st.session_state["current_analysis"] = None
        st.session_state["active_context_signature"] = context_sig

    if audio_bytes is not None:
        st.markdown(f"**Audio Terpilih ({source_label}):**")
        st.audio(audio_bytes)

    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    with col_btn1:
        analyze_clicked = st.button(
            "Jalankan Analisis",
            type="primary",
            disabled=(audio_bytes is None),
        )
    with col_btn2:
        if st.button("Rekam Ulang"):
            for k in (f"demo_audio_bytes_{mode}", f"active_input_type_{mode}", f"last_seen_mic_{mode}", f"last_seen_upload_{mode}"):
                if k in st.session_state:
                    del st.session_state[k]
            clear_current_input_and_result()
            st.rerun()
    with col_btn3:
        if st.button("Reset Sesi"):
            for k in (f"demo_audio_bytes_{mode}", f"active_input_type_{mode}", f"last_seen_mic_{mode}", f"last_seen_upload_{mode}"):
                if k in st.session_state:
                    del st.session_state[k]
            reset_entire_session()
            st.rerun()

    return audio_bytes, filename_hint, analyze_clicked


def save_attempt_to_session(attempt_record: Dict[str, Any]) -> int:
    """Simpan hasil analisis yang valid sebagai percobaan baru di dalam sesi aktif."""
    init_session_state()
    attempts = st.session_state["attempts"]
    attempt_number = len(attempts) + 1
    attempt_record["attempt_id"] = f"Percobaan #{attempt_number}"
    attempts.append(attempt_record)
    return attempt_number
