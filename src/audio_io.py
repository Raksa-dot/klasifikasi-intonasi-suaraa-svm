"""Modul pembacaan dan penulisan input audio (WAV, MP3, FLAC, OGG) untuk website dan training."""

import io
from pathlib import Path
import tempfile
from typing import Tuple, Union
import librosa
import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf


def read_audio_bytes(audio_bytes: bytes, filename_hint: str = "input.wav") -> Tuple[np.ndarray, int]:
    """Baca byte file audio menggunakan SoundFile, dengan fallback Scipy WAV dan Librosa."""
    if not audio_bytes:
        raise ValueError("Data byte audio kosong.")

    # 1. Coba baca langsung menggunakan soundfile
    try:
        with io.BytesIO(audio_bytes) as buffer:
            samples, sample_rate = sf.read(buffer, dtype="float32", always_2d=False)
            return np.asarray(samples, dtype=np.float32), int(sample_rate)
    except Exception:
        pass

    # 2. Coba fallback menggunakan scipy.io.wavfile (khusus format WAV murni)
    try:
        with io.BytesIO(audio_bytes) as buffer:
            sr, data = wavfile.read(buffer)
            # Normalisasi tipe data ke float32 [-1.0, 1.0]
            if data.dtype == np.int16:
                samples = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                samples = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                samples = (data.astype(np.float32) - 128.0) / 128.0
            else:
                samples = data.astype(np.float32)
            return np.asarray(samples, dtype=np.float32), int(sr)
    except Exception:
        pass

    # 3. Deteksi format dari magic bytes file
    suffix = ".wav"
    if audio_bytes.startswith(b"ID3") or audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        suffix = ".mp3"
    elif audio_bytes.startswith(b"OggS"):
        suffix = ".ogg"
    elif audio_bytes.startswith(b"fLaC"):
        suffix = ".flac"
    elif audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
        suffix = ".webm"
    else:
        hint_sfx = Path(filename_hint).suffix.lower()
        if hint_sfx in (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"):
            suffix = hint_sfx

    # 4. Fallback ke librosa melalui temporary file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = Path(tmp_file.name)
        samples, sample_rate = librosa.load(str(tmp_path), sr=None, mono=False)
        if samples.ndim == 2:
            samples = samples.T
        return np.asarray(samples, dtype=np.float32), int(sample_rate)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def read_audio_file(file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """Baca file audio dari path disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File audio tidak ditemukan: {path}")
    try:
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        return np.asarray(samples, dtype=np.float32), int(sample_rate)
    except Exception:
        samples, sample_rate = librosa.load(str(path), sr=None, mono=False)
        if samples.ndim == 2:
            samples = samples.T
        return np.asarray(samples, dtype=np.float32), int(sample_rate)


def encode_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Konversi array sinyal audio menjadi byte WAV PCM untuk pemutar audio browser."""
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(samples, dtype=np.float32), int(sample_rate), format="WAV")
    return buffer.getvalue()
