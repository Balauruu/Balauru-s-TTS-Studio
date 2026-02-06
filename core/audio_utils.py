import os
import tempfile
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment


def generate_silence(duration_seconds: float, sample_rate: int = 24000) -> np.ndarray:
    """Generate a silence array of the given duration."""
    return np.zeros(int(duration_seconds * sample_rate), dtype=np.float32)


def concat_audio(segments: list[np.ndarray]) -> np.ndarray:
    """Concatenate a list of audio arrays into one."""
    if not segments:
        return np.array([], dtype=np.float32)
    return np.concatenate(segments)


def normalize_loudness(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize audio to target amplitude."""
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to a different sample rate."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Change playback speed without changing pitch."""
    if rate == 1.0:
        return audio
    return librosa.effects.time_stretch(audio, rate=rate)


def save_audio(audio: np.ndarray, path: str, sample_rate: int,
               output_format: str = "wav") -> str:
    """Save audio array to file in the specified format.

    Returns the final file path (extension may differ from input path).
    """
    base, _ = os.path.splitext(path)
    final_path = f"{base}.{output_format}"

    if output_format == "wav":
        sf.write(final_path, audio, sample_rate, subtype="PCM_16")
    elif output_format == "flac":
        sf.write(final_path, audio, sample_rate, format="FLAC")
    elif output_format == "mp3":
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            sf.write(tmp_wav, audio, sample_rate, subtype="PCM_16")
            AudioSegment.from_wav(tmp_wav).export(final_path, format="mp3", bitrate="320k")
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)
    else:
        sf.write(final_path, audio, sample_rate, subtype="PCM_16")

    return final_path


def load_audio(path: str, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load an audio file, optionally resampling.

    Returns (audio_array, sample_rate).
    """
    audio, orig_sr = librosa.load(path, sr=sr)
    return audio, sr if sr else orig_sr
