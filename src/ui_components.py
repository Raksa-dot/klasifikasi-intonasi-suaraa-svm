"""Komponen UI bersama Streamlit: Input Audio (Mikrofon & Upload), Manajemen State Sesi, dan Live Pitch Monitor."""

import hashlib
from typing import Any, Dict, Optional, Tuple
import streamlit as st
import streamlit.components.v1 as components

from src.config import DATA_DIR, MAX_DURATION_SEC, MAX_UPLOAD_SIZE_MB, MIN_DURATION_SEC


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
    """Tampilkan widget Live Pitch Monitor / Tuner Real-Time berbasis Web Audio API di browser."""
    target_hz_js = f"{target_hz:.4f}" if (target_hz is not None and target_hz > 0) else "null"
    mode_title = (
        f"Live Pitch & Pola Intonasi Monitor (Target: {target_label})"
        if mode == "SPEAKING"
        else f"Live Pitch Tuner Real-Time (Target: {target_label} — {target_hz:.2f} Hz)"
        if target_hz
        else "Live Pitch Monitor Real-Time"
    )

    html_code = f"""
    <div style="font-family: system-ui, -apple-system, sans-serif; border: 1px solid #cbd5e1; border-radius: 10px; padding: 14px; background: #f8fafc;">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">
        <div>
          <strong style="color: #0f172a; font-size: 15px;">⚡ {mode_title}</strong>
          <div style="color: #475569; font-size: 12px;">Visualisasi pitch langsung di browser saat Anda bersuara di mikrofon (panduan langsung sebelum/sambil latihan).</div>
        </div>
        <div>
          <button id="btnStartLive" style="background: #0f766e; color: white; border: none; border-radius: 6px; padding: 7px 14px; font-weight: 600; cursor: pointer; font-size: 13px;">🎙️ Mulai Monitor Live</button>
          <button id="btnStopLive" style="background: #dc2626; color: white; border: none; border-radius: 6px; padding: 7px 14px; font-weight: 600; cursor: pointer; font-size: 13px; display: none;">⏹️ Hentikan Monitor</button>
        </div>
      </div>

      <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 10px; text-align: center;">
        <div style="background: white; padding: 8px; border-radius: 8px; border: 1px solid #e2e8f0;">
          <div style="font-size: 11px; color: #64748b;">Pitch / F0 Saat Ini</div>
          <div id="liveHz" style="font-size: 18px; font-weight: 700; color: #0f172a;">— Hz</div>
        </div>
        <div style="background: white; padding: 8px; border-radius: 8px; border: 1px solid #e2e8f0;">
          <div style="font-size: 11px; color: #64748b;">Nada Terdekat</div>
          <div id="liveNote" style="font-size: 18px; font-weight: 700; color: #0f766e;">—</div>
        </div>
        <div style="background: white; padding: 8px; border-radius: 8px; border: 1px solid #e2e8f0;">
          <div style="font-size: 11px; color: #64748b;">Selisih (Cents)</div>
          <div id="liveCents" style="font-size: 18px; font-weight: 700; color: #2563eb;">—</div>
        </div>
        <div style="background: white; padding: 8px; border-radius: 8px; border: 1px solid #e2e8f0;">
          <div style="font-size: 11px; color: #64748b;">Tren Kontur Live</div>
          <div id="livePattern" style="font-size: 14px; font-weight: 700; color: #7c3aed;">Menunggu suara...</div>
        </div>
      </div>

      <canvas id="pitchCanvas" width="760" height="145" style="width: 100%; height: 145px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;"></canvas>
      <div id="liveStatus" style="font-size: 11px; color: #64748b; margin-top: 6px;">
        Klik "Mulai Monitor Live" untuk melihat pergerakan pitch secara real-time, lalu gunakan perekam di bawah untuk analisis penuh Python (librosa.pyin + SVM).
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

        const c = new Array(size).fill(0);
        for (let i = 0; i < size; i++) {{
          for (let j = 0; j < size - i; j++) {{
            c[i] = c[i] + buf[j] * buf[j + i];
          }}
        }}
        let d = 0;
        while (d < size - 1 && c[d] > c[d + 1]) d++;
        let maxval = -1, maxpos = -1;
        for (let i = d; i < size; i++) {{
          if (c[i] > maxval) {{
            maxval = c[i];
            maxpos = i;
          }}
        }}
        let T0 = maxpos;
        if (T0 <= 0) return -1;
        const f0 = sampleRate / T0;
        if (f0 < 65 || f0 > 950) return -1;
        return f0;
      }}

      function estimateLiveTrend(arr) {{
        const valid = arr.filter(v => v !== null && v > 0);
        if (valid.length < 12) return "Mengumpulkan frame...";
        const n = valid.length;
        const third = Math.max(2, Math.floor(n / 3));
        const avg = a => a.reduce((s, x) => s + x, 0) / a.length;
        const s1 = 12 * Math.log2(avg(valid.slice(0, third)) / 220);
        const s2 = 12 * Math.log2(avg(valid.slice(third, 2 * third)) / 220);
        const s3 = 12 * Math.log2(avg(valid.slice(n - third)) / 220);
        const d12 = s2 - s1;
        const d23 = s3 - s2;
        const d13 = s3 - s1;
        if (d12 < -0.55 && d23 > 0.55) return "FALL_RISE (Turun→Naik)";
        if (d12 > 0.55 && d23 < -0.55) return "RISE_FALL (Naik→Turun)";
        if (d13 >= 0.85) return "RISING (Cenderung Naik)";
        if (d13 <= -0.85) return "FALLING (Cenderung Turun)";
        return "LEVEL (Relatif Datar)";
      }}

      function drawCanvas() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "#f1f5f9";
        ctx.lineWidth = 1;
        for (let g = 1; g < 4; g++) {{
          const y = (canvas.height / 4) * g;
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(canvas.width, y);
          ctx.stroke();
        }}

        const valid = historyHz.filter(v => v !== null && v > 0);
        if (valid.length === 0) return;

        let minF = Math.min(...valid);
        let maxF = Math.max(...valid);
        if (targetHz) {{
          minF = Math.min(minF, targetHz * 0.85);
          maxF = Math.max(maxF, targetHz * 1.15);
        }}
        const pad = Math.max(25, (maxF - minF) * 0.2);
        minF = Math.max(50, minF - pad);
        maxF = maxF + pad;

        if (targetHz) {{
          const ty = canvas.height - ((targetHz - minF) / (maxF - minF)) * (canvas.height - 20) - 10;
          ctx.strokeStyle = "#dc2626";
          ctx.setLineDash([5, 5]);
          ctx.beginPath();
          ctx.moveTo(0, ty);
          ctx.lineTo(canvas.width, ty);
          ctx.stroke();
          ctx.setLineDash([]);
        }}

        ctx.strokeStyle = "#0f766e";
        ctx.lineWidth = 2.5;
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

        if (f0 !== -1) {{
          historyHz.push(f0);
          const midi = 69 + 12 * Math.log2(f0 / 440.0);
          const nearestMidi = Math.round(midi);
          const noteName = NOTE_STRINGS[(nearestMidi % 12 + 12) % 12] + (Math.floor(nearestMidi / 12) - 1);
          const refFreq = targetHz ? targetHz : 440.0 * Math.pow(2, (nearestMidi - 69) / 12.0);
          const cents = 1200 * Math.log2(f0 / refFreq);

          elHz.textContent = f0.toFixed(1) + " Hz";
          elNote.textContent = noteName;
          elCents.textContent = (cents >= 0 ? "+" : "") + cents.toFixed(1) + " c";
          elPattern.textContent = estimateLiveTrend(historyHz.slice(-45));
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
          elStatus.textContent = "Mikrofon live aktif. Bersuaralah untuk melihat grafik pitch dan estimasi tren kontur secara langsung.";
          updatePitch();
        }} catch (err) {{
          elStatus.textContent = "Akses mikrofon ditolak atau tidak tersedia di browser (" + err.message + "). Gunakan opsi Unggah Audio di bawah.";
        }}
      }});

      btnStop.addEventListener("click", () => {{
        if (rafId) cancelAnimationFrame(rafId);
        if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
        if (audioCtx) audioCtx.close();
        analyser = null;
        btnStop.style.display = "none";
        btnStart.style.display = "inline-block";
        elStatus.textContent = "Monitor live dihentikan. Silakan rekam atau unggah audio di bawah untuk analisis penuh.";
      }});
    }})();
    </script>
    """
    components.html(html_code, height=315)


def render_shared_audio_input(
    mode: str,
    exercise_id: str,
    target_key: str,
) -> Tuple[Optional[bytes], str, bool]:
    """Tampilkan komponen input audio bersama (mikrofon browser, unggah file, & sampel demo) serta sinkronisasi state.

    Mengembalikan (audio_bytes, filename_hint, analyze_clicked).
    Saat audio, mode, atau target berubah, hasil analisis lama dibatalkan otomatis.
    """
    init_session_state()
    reset_cnt = st.session_state.get("input_reset_counter", 0)

    st.markdown("#### 🎙️ Input Rekaman Audio Latihan")
    st.caption(
        f"**Batas Konfigurasi Input:** Format **WAV** (atau MP3/FLAC/OGG), durasi **{MIN_DURATION_SEC:.1f} – {MAX_DURATION_SEC:.1f} detik** "
        f"(disesuaikan untuk satu frasa/nada latihan pendek), ukuran maksimum **{MAX_UPLOAD_SIZE_MB} MB**."
    )

    tab_mic, tab_upload, tab_demo = st.tabs([
        "🎙️ Rekam via Mikrofon Browser",
        "📂 Unggah File Audio (WAV/MP3)",
        "✨ Contoh Audio Uji (Demo Langsung)",
    ])

    audio_bytes: Optional[bytes] = None
    filename_hint = "recording.wav"

    with tab_mic:
        st.markdown(
            """
            **Panduan Merekam Suara:**
            1. Tekan ikon **Mikrofon** di bawah untuk memulai perekaman.
            2. Ucapkan frasa atau nyanyikan nada sesuai instruksi target latihan.
            3. Tekan tombol **Stop** (lingkaran merah / centang) setelah selesai merekam.
            """
        )
        mic_audio = st.audio_input(
            "Perekam Suara Mikrofon Langsung:",
            key=f"mic_{mode}_{reset_cnt}",
            help="Klik ikon mikrofon untuk merekam audio langsung dari mikrofon browser Anda.",
        )
        if mic_audio is not None:
            audio_bytes = mic_audio.getvalue()
            filename_hint = getattr(mic_audio, "name", "mic_recording.wav") or "mic_recording.wav"
            st.success("✅ Rekaman mikrofon berhasil ditangkap! Dengarkan pemutar di bawah lalu klik **🔍 Analisis Rekaman Suara**.")
        else:
            st.info("💡 **Petunjuk Mikrofon:** Jika browser meminta izin mikrofon, pilih **Allow / Izinkan**.")

    with tab_upload:
        st.markdown("**Unggah File Audio dari Komputer/Ponsel Anda:**")
        uploaded_file = st.file_uploader(
            "Pilih file audio WAV (opsional MP3/FLAC/OGG):",
            type=["wav", "mp3", "flac", "ogg"],
            key=f"upload_{mode}_{reset_cnt}",
        )
        if uploaded_file is not None and audio_bytes is None:
            audio_bytes = uploaded_file.getvalue()
            filename_hint = uploaded_file.name or "uploaded.wav"
            st.success(f"✅ File audio `{filename_hint}` ({len(audio_bytes)/1024:.1f} KB) berhasil diunggah!")

    with tab_demo:
        st.markdown("**Gunakan Sampel Rekaman Nyata untuk Pengujian Cepat:**")
        st.caption("Pilihan ini sangat berguna jika Anda ingin menguji sistem analisis secara instan tanpa merekam suara.")

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
                "Pilih contoh audio dari dataset:",
                range(len(demo_files)),
                format_func=lambda i: f"Sampel {i+1}: {demo_files[i][0]}",
                key=f"demo_sel_{mode}_{reset_cnt}",
            )
            sel_path = demo_files[demo_idx][1]
            if st.button("▶️ Gunakan Sampel Audio Ini Sebagai Input", key=f"btn_demo_{mode}_{reset_cnt}"):
                st.session_state[f"demo_audio_bytes_{mode}"] = sel_path.read_bytes()
                st.session_state[f"demo_audio_name_{mode}"] = demo_files[demo_idx][0]
                st.rerun()

        demo_stored = st.session_state.get(f"demo_audio_bytes_{mode}")
        if demo_stored and audio_bytes is None:
            audio_bytes = demo_stored
            filename_hint = st.session_state.get(f"demo_audio_name_{mode}", "demo_audio.wav")
            st.success(f"✅ Contoh audio `{filename_hint}` aktif sebagai input.")

    # Buat tanda tangan unik dari (mode, exercise_id, target_key, hash(audio_bytes))
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()[:16] if audio_bytes else "no_audio"
    context_sig = f"{mode}|{exercise_id}|{target_key}|{audio_hash}"

    # Batalkan hasil lama secara otomatis jika input audio, mode, atau target berubah
    if st.session_state.get("active_context_signature") != context_sig:
        st.session_state["current_analysis"] = None
        st.session_state["active_context_signature"] = context_sig

    if audio_bytes is not None:
        st.markdown("**🎧 Pemutar Audio Siap Dianalisis:**")
        st.audio(audio_bytes)

    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    with col_btn1:
        analyze_clicked = st.button(
            "🔍 Analisis Rekaman Suara",
            type="primary",
            disabled=(audio_bytes is None),
        )
    with col_btn2:
        if st.button("🔄 Rekam Ulang / Ganti Audio"):
            if f"demo_audio_bytes_{mode}" in st.session_state:
                del st.session_state[f"demo_audio_bytes_{mode}"]
            clear_current_input_and_result()
            st.rerun()
    with col_btn3:
        if st.button("🗑️ Reset Seluruh Sesi"):
            if f"demo_audio_bytes_{mode}" in st.session_state:
                del st.session_state[f"demo_audio_bytes_{mode}"]
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
