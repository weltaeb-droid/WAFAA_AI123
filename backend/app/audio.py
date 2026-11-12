"""Audio decoding and educational signal analysis.

Everything that reaches the model or the charts passes through this module:

* the uploaded/recorded clip is written to a temporary file with an allow-listed
  extension,
* ``ffmpeg`` decodes it to **mono, 16 kHz, 16-bit PCM** (the contract the
  WAFAA AI model was trained on),
* the resulting samples are analysed for waveform, spectrum, spectrogram,
  dominant frequency and frequency distribution.

The measurements describe *audio characteristics only*. They are not, and must
never be presented as, a medical interpretation of a baby's cry.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


class AudioError(Exception):
    """Raised for any user-correctable audio problem (bad format, too long...)."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(slots=True)
class DecodedAudio:
    samples: np.ndarray  # float32, mono, [-1, 1]
    sample_rate: int
    duration_seconds: float
    source_extension: str


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def sanitize_extension(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in config.ALLOWED_AUDIO_EXTENSIONS:
        raise AudioError(
            "unsupported_audio_format",
            "Unsupported audio format. Please use WAV, MP3, M4A, OGG, OPUS, "
            "FLAC or WebM audio.",
            415,
        )
    return suffix


def decode_audio(raw: bytes, filename: str | None) -> DecodedAudio:
    """Decode any supported upload into mono float32 samples at 16 kHz."""
    if not raw:
        raise AudioError("empty_file", "The uploaded file is empty.")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise AudioError(
            "file_too_large",
            f"The file is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            413,
        )
    if not ffmpeg_available():
        raise AudioError(
            "decoder_unavailable",
            "The audio decoder (ffmpeg) is not available on the server.",
            503,
        )

    extension = sanitize_extension(filename)
    with tempfile.TemporaryDirectory(prefix="wafaa-audio-") as tmp:
        source = Path(tmp) / f"input{extension}"
        source.write_bytes(raw)
        target = Path(tmp) / "decoded.wav"
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            str(config.TARGET_CHANNELS),
            "-ar",
            str(config.TARGET_SAMPLE_RATE),
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            str(target),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                timeout=90,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - defensive
            raise AudioError(
                "decode_timeout", "Audio decoding took too long.", 422
            ) from exc
        if completed.returncode != 0 or not target.exists():
            raise AudioError(
                "decode_failed",
                "This file could not be decoded as audio. It may be corrupted or "
                "in an unsupported format.",
                422,
            )
        samples = _read_pcm_wav(target)

    duration = len(samples) / config.TARGET_SAMPLE_RATE
    if duration < config.MIN_AUDIO_SECONDS:
        raise AudioError(
            "audio_too_short",
            "The clip is too short. Please record at least "
            f"{int(config.MIN_AUDIO_SECONDS)} second of audio.",
            422,
        )
    if duration > config.MAX_AUDIO_SECONDS:
        raise AudioError(
            "audio_too_long",
            "The clip is too long. Please use a recording of "
            f"{int(config.MAX_AUDIO_SECONDS)} seconds or less.",
            422,
        )
    return DecodedAudio(
        samples=samples,
        sample_rate=config.TARGET_SAMPLE_RATE,
        duration_seconds=duration,
        source_extension=extension,
    )


def _read_pcm_wav(path: Path) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.readframes(handle.getnframes())
            channels = handle.getnchannels()
            width = handle.getsampwidth()
    except (wave.Error, OSError) as exc:
        raise AudioError(
            "decode_failed", "The decoded audio could not be read.", 422
        ) from exc

    if width != 2:
        raise AudioError("decode_failed", "Unexpected audio sample format.", 422)
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)


def write_pcm16_wav(samples: np.ndarray, path: Path, sample_rate: int) -> None:
    """Persist a decoded clip as 16-bit mono WAV (used only with consent)."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


# --- analysis helpers --------------------------------------------------------

_FRAME_SIZE = 1024


def _to_db(value: float, floor: float = -100.0) -> float:
    if value <= 0:
        return floor
    return max(floor, 20.0 * math.log10(value))


def _frame_signal(samples: np.ndarray, frame: int) -> np.ndarray:
    if len(samples) <= frame:
        padded = np.zeros(frame, dtype=np.float32)
        padded[: len(samples)] = samples
        return padded.reshape(1, frame)
    usable = (len(samples) // frame) * frame
    return samples[:usable].reshape(-1, frame)


def _quality_report(samples: np.ndarray) -> dict:
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if len(samples) else 0.0
    clipping_ratio = float(np.mean(np.abs(samples) > 0.985)) if len(samples) else 0.0
    frames = _frame_signal(samples, _FRAME_SIZE)
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    silence_threshold = max(0.0015, peak * 0.02)
    silence_ratio = float(np.mean(frame_rms < silence_threshold)) if len(frame_rms) else 1.0

    reasons: list[str] = []
    if clipping_ratio > 0.01:
        reasons.append("clipping")
    if _to_db(rms) < -42.0:
        reasons.append("too_quiet")
    if silence_ratio > 0.85:
        reasons.append("mostly_silence")
    if len(samples) / config.TARGET_SAMPLE_RATE < 2.0:
        reasons.append("short_clip")

    if len(reasons) == 0:
        level = "good"
    elif len(reasons) == 1 and reasons[0] == "short_clip":
        level = "fair"
    elif len(reasons) <= 2:
        level = "fair"
    else:
        level = "poor"

    return {
        "level": level,
        "reasons": reasons,
        "peak_dbfs": round(_to_db(peak), 2),
        "rms_dbfs": round(_to_db(rms), 2),
        "clipping_ratio": round(clipping_ratio, 5),
        "silence_ratio": round(silence_ratio, 4),
    }


def _waveform_envelope(samples: np.ndarray, buckets: int = 320) -> list[float]:
    if len(samples) == 0:
        return [0.0] * buckets
    usable = min(buckets, len(samples))
    edges = np.linspace(0, len(samples), usable + 1, dtype=np.int64)
    envelope: list[float] = []
    for index in range(usable):
        start, end = edges[index], edges[index + 1]
        chunk = samples[start:end]
        if len(chunk) == 0:
            envelope.append(0.0)
            continue
        # Signed extreme keeps the waveform shape readable.
        index_max = int(np.argmax(np.abs(chunk)))
        envelope.append(round(float(chunk[index_max]), 4))
    if usable < buckets:
        envelope.extend([0.0] * (buckets - usable))
    return envelope


def _average_power_spectrum(samples: np.ndarray, n_fft: int = 1024) -> np.ndarray:
    frames = _frame_signal(samples, n_fft)
    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    return spectrum.mean(axis=0)


def _downsample_series(values: np.ndarray, target: int) -> np.ndarray:
    if len(values) <= target:
        return values
    edges = np.linspace(0, len(values), target + 1, dtype=np.int64)
    return np.array(
        [float(values[a:b].mean()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    )


def _dominant_frequency(power: np.ndarray, sample_rate: int, n_fft: int) -> float:
    if len(power) < 3 or float(power.sum()) <= 0.0:
        return 0.0
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    # Ignore DC / rumble below 40 Hz which is rarely the audible peak.
    valid = freqs >= 40.0
    if not np.any(valid):
        return 0.0
    masked = np.where(valid, power, 0.0)
    peak_index = int(np.argmax(masked))
    # Parabolic interpolation around the peak for a smoother estimate.
    if 0 < peak_index < len(masked) - 1:
        left, center, right = (
            float(masked[peak_index - 1]),
            float(masked[peak_index]),
            float(masked[peak_index + 1]),
        )
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-12:
            offset = 0.5 * (left - right) / denominator
            offset = max(-0.5, min(0.5, offset))
            return float(freqs[peak_index] + offset * (sample_rate / n_fft))
    return float(freqs[peak_index])


def _spectrum_curve(power: np.ndarray, sample_rate: int, n_fft: int, points: int = 160) -> dict:
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    upper = min(sample_rate / 2.0, 8000.0)
    mask = freqs <= upper
    freqs = freqs[mask]
    power = power[mask]
    if len(power) == 0:
        return {"frequencies": [], "magnitudes": []}
    freqs = _downsample_series(freqs, points)
    power = _downsample_series(power, points)
    reference = float(power.max()) if power.size else 0.0
    if reference <= 0:
        magnitudes = np.zeros_like(power)
    else:
        magnitudes = np.clip(
            (10.0 * np.log10(np.maximum(power, 1e-12)) - 10.0 * math.log10(reference) + 70.0)
            / 70.0,
            0.0,
            1.0,
        )
    return {
        "frequencies": [round(float(value), 1) for value in freqs],
        "magnitudes": [round(float(value), 4) for value in magnitudes],
    }


def _spectrogram(samples: np.ndarray, sample_rate: int) -> dict:
    n_fft = 512
    hop = 256
    window = np.hanning(n_fft).astype(np.float32)
    if len(samples) < n_fft:
        samples = np.pad(samples, (0, n_fft - len(samples)))
    frame_count = 1 + (len(samples) - n_fft) // hop
    frame_count = max(1, min(frame_count, 600))
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(frame_count, n_fft),
        strides=(samples.strides[0] * hop, samples.strides[0]),
        writeable=False,
    )
    magnitude = np.abs(np.fft.rfft(frames * window, axis=1))
    # Keep 0-4000 Hz: the informative band for infant cries at 16 kHz.
    max_bin = int(4000.0 / (sample_rate / n_fft)) + 1
    magnitude = magnitude[:, : max_bin]
    db = 20.0 * np.log10(np.maximum(magnitude, 1e-8))
    db -= db.max() if db.size else 0.0
    normalized = np.clip((db + 70.0) / 70.0, 0.0, 1.0)

    time_bins = min(120, normalized.shape[0])
    freq_bins = min(64, normalized.shape[1])
    values = np.zeros((time_bins, freq_bins), dtype=np.float32)
    t_edges = np.linspace(0, normalized.shape[0], time_bins + 1, dtype=np.int64)
    f_edges = np.linspace(0, normalized.shape[1], freq_bins + 1, dtype=np.int64)
    for ti in range(time_bins):
        row = normalized[t_edges[ti] : max(t_edges[ti] + 1, t_edges[ti + 1])]
        row = row.mean(axis=0) if len(row) else np.zeros(normalized.shape[1])
        for fi in range(freq_bins):
            values[ti, fi] = row[f_edges[fi] : max(f_edges[fi] + 1, f_edges[fi + 1])].mean()

    frequencies = [
        round(float((f_edges[fi] + f_edges[fi + 1]) / 2 * (sample_rate / n_fft)), 1)
        for fi in range(freq_bins)
    ]
    return {
        "frames": time_bins,
        "bins": freq_bins,
        "frequencies": frequencies,
        "values": [[round(float(v), 3) for v in row] for row in values],
        "max_frequency_hz": 4000.0,
    }


_BANDS: list[tuple[str, float, float]] = [
    ("0-250", 0.0, 250.0),
    ("250-500", 250.0, 500.0),
    ("500-1000", 500.0, 1000.0),
    ("1000-2000", 1000.0, 2000.0),
    ("2000-4000", 2000.0, 4000.0),
    ("4000-8000", 4000.0, 8000.0),
]


def _frequency_distribution(power: np.ndarray, sample_rate: int, n_fft: int) -> list[dict]:
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    total = float(power.sum())
    distribution: list[dict] = []
    for key, low, high in _BANDS:
        mask = (freqs >= low) & (freqs < high)
        energy = float(power[mask].sum()) if np.any(mask) else 0.0
        share = (energy / total) if total > 0 else 0.0
        distribution.append(
            {
                "key": key,
                "from_hz": low,
                "to_hz": high,
                "share": round(share, 4),
            }
        )
    return distribution


def analyse_signal(decoded: DecodedAudio) -> dict:
    """Build the full educational audio report for a decoded clip."""
    samples = decoded.samples
    sample_rate = decoded.sample_rate
    n_fft = 1024
    power = _average_power_spectrum(samples, n_fft)
    return {
        "duration_seconds": round(decoded.duration_seconds, 3),
        "sample_rate": sample_rate,
        "channels": config.TARGET_CHANNELS,
        "source_extension": decoded.source_extension,
        "quality": _quality_report(samples),
        "waveform": _waveform_envelope(samples),
        "spectrum": _spectrum_curve(power, sample_rate, n_fft),
        "spectrogram": _spectrogram(samples, sample_rate),
        "dominant_frequency_hz": round(_dominant_frequency(power, sample_rate, n_fft), 1),
        "frequency_bands": _frequency_distribution(power, sample_rate, n_fft),
        "loudness_dbfs": round(
            _to_db(
                float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
                if len(samples)
                else 0.0
            ),
            2,
        ),
    }
