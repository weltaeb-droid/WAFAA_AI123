"""Feature extraction for the WAFAA AI cry classifier.

The API must feed the trained model exactly the features it saw during
training. The canonical pipeline implemented here mirrors the widely used
``librosa.feature.mfcc`` defaults, but is written with NumPy only so the server
image stays small and installs everywhere (including Alpine/musl):

======================  ==========================================
Parameter               Value
======================  ==========================================
sample rate             16 000 Hz, mono
n_fft                   2048
hop_length              512
window                  Hann
spectrogram             power ``|STFT|^2``
mel filters             128, Slaney scale, fmin 0, fmax 8000
log compression         10*log10, floor -100 dB (librosa-style top_db=80)
DCT                     type II, orthonormal, first 40 coefficients
feature vector          mean+std of 40 MFCCs (80)
                        mean+std of 40 delta-MFCCs (80)
                        10 scalar descriptors (mean+std of spectral
                        centroid, bandwidth, rolloff, flatness and RMS)
total dimension         170
======================  ==========================================

``FEATURE_VECTOR_VERSION`` is written next to every analysis result and can be
compared with the value declared in the model sidecar file
(``wafaa_cry_model.meta.json``). ``scripts/check_feature_compat.py`` compares
this implementation against librosa when librosa is installed.
"""

from __future__ import annotations

import math

import numpy as np

from . import config

FEATURE_VECTOR_VERSION = "wafaa-mfcc-v1"
FEATURE_DIMENSION = 170

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 40
FMIN = 0.0
FMAX = 8000.0
LOG_FLOOR_DB = -100.0

_EPS = 1e-10


def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    """Slaney mel scale (matches librosa's default htk=False)."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    f_min, f_sp = 0.0, 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    log_region = frequencies >= min_log_hz
    mels[log_region] = min_log_mel + np.log(frequencies[log_region] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    f_min, f_sp = 0.0, 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    freqs = f_min + f_sp * mels
    log_region = mels >= min_log_mel
    freqs[log_region] = min_log_hz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freqs


def mel_filterbank(sample_rate: int = config.TARGET_SAMPLE_RATE, n_fft: int = N_FFT,
                   n_mels: int = N_MELS, fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """Slaney-normalised triangular mel filterbank of shape (n_mels, 1 + n_fft/2)."""
    fft_frequencies = np.linspace(0.0, sample_rate / 2.0, 1 + n_fft // 2)
    mel_edges = np.linspace(_hz_to_mel(np.array([fmin]))[0], _hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)
    ramps = hz_edges[:, None] - fft_frequencies[None, :]
    weights = np.zeros((n_mels, len(fft_frequencies)), dtype=np.float64)
    for index in range(n_mels):
        lower = -ramps[index] / max(hz_edges[index + 1] - hz_edges[index], _EPS)
        upper = ramps[index + 2] / max(hz_edges[index + 2] - hz_edges[index + 1], _EPS)
        weights[index] = np.maximum(0.0, np.minimum(lower, upper))
    # Slaney area normalisation.
    widths = np.diff(hz_edges)
    weights *= (2.0 / np.maximum(widths[:, None], _EPS))
    return weights


def dct_matrix(n_out: int, n_in: int) -> np.ndarray:
    """Orthonormal DCT-II matrix of shape (n_out, n_in)."""
    samples = np.arange(n_in, dtype=np.float64)
    basis = np.cos(np.pi / n_in * (samples[None, :] + 0.5) * np.arange(n_out)[:, None])
    basis[0, :] *= 1.0 / math.sqrt(2.0)
    return basis * math.sqrt(2.0 / n_in)


def _stft_power(samples: np.ndarray, n_fft: int = N_FFT, hop: int = HOP_LENGTH) -> np.ndarray:
    if len(samples) < n_fft:
        samples = np.pad(samples, (0, n_fft - len(samples)))
    frames = 1 + (len(samples) - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    view = np.lib.stride_tricks.as_strided(
        samples,
        shape=(frames, n_fft),
        strides=(samples.strides[0] * hop, samples.strides[0]),
        writeable=False,
    )
    return np.abs(np.fft.rfft(view * window, axis=1)) ** 2


def _delta(features: np.ndarray, width: int = 9) -> np.ndarray:
    """Standard regression delta (same definition librosa uses by default)."""
    if features.shape[0] < 2:
        return np.zeros_like(features)
    half = width // 2
    padded = np.pad(features, ((half, half), (0, 0)), mode="edge")
    denominator = 2.0 * sum(index * index for index in range(1, half + 1))
    deltas = np.zeros_like(features)
    for index in range(1, half + 1):
        deltas += index * (padded[half + index : half + index + features.shape[0]]
                           - padded[half - index : half - index + features.shape[0]])
    return deltas / denominator


def _spectral_descriptors(power: np.ndarray, sample_rate: int = config.TARGET_SAMPLE_RATE) -> dict[str, np.ndarray]:
    freqs = np.fft.rfftfreq(power.shape[1] * 2 - 1, d=1.0 / sample_rate)[: power.shape[1]]
    magnitudes = np.sqrt(np.maximum(power, 0.0))
    total = np.maximum(magnitudes.sum(axis=1), _EPS)
    centroid = (magnitudes * freqs[None, :]).sum(axis=1) / total
    bandwidth = np.sqrt(
        (magnitudes * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / total
    )
    cumulative = np.cumsum(magnitudes, axis=1)
    rolloff_threshold = 0.85 * total
    rolloff_index = np.argmax(cumulative >= rolloff_threshold[:, None], axis=1)
    rolloff = freqs[rolloff_index]
    flatness = np.exp(np.mean(np.log(np.maximum(magnitudes, _EPS)), axis=1)) / np.maximum(
        np.mean(magnitudes, axis=1), _EPS
    )
    rms = np.sqrt(np.mean(power, axis=1))
    return {
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "flatness": flatness,
        "rms": rms,
    }


def extract_features(samples: np.ndarray, sample_rate: int = config.TARGET_SAMPLE_RATE) -> np.ndarray:
    """Return the 170-dimensional float32 feature vector for one clip."""
    if samples.size == 0:
        return np.zeros(FEATURE_DIMENSION, dtype=np.float32)
    samples = np.ascontiguousarray(samples, dtype=np.float32)

    power = _stft_power(samples)
    mel_basis = mel_filterbank(sample_rate=sample_rate, fmax=min(FMAX, sample_rate / 2.0))
    mel_power = power @ mel_basis.T
    mel_db = 10.0 * np.log10(np.maximum(mel_power, _EPS))
    mel_db = np.maximum(mel_db, mel_db.max() - 80.0)
    mfcc = mel_db @ dct_matrix(N_MFCC, mel_basis.shape[0]).T
    delta = _delta(mfcc)

    descriptors = _spectral_descriptors(power, sample_rate)
    scalars = np.concatenate(
        [
            [descriptors["centroid"].mean(), descriptors["centroid"].std()],
            [descriptors["bandwidth"].mean(), descriptors["bandwidth"].std()],
            [descriptors["rolloff"].mean(), descriptors["rolloff"].std()],
            [descriptors["flatness"].mean(), descriptors["flatness"].std()],
            [descriptors["rms"].mean(), descriptors["rms"].std()],
        ]
    )

    vector = np.concatenate(
        [
            mfcc.mean(axis=0),
            mfcc.std(axis=0),
            delta.mean(axis=0),
            delta.std(axis=0),
            scalars,
        ]
    )
    if vector.shape[0] != FEATURE_DIMENSION:  # pragma: no cover - guard
        raise RuntimeError(
            f"Feature vector has {vector.shape[0]} values, expected {FEATURE_DIMENSION}."
        )
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def feature_metadata() -> dict:
    return {
        "version": FEATURE_VECTOR_VERSION,
        "dimension": FEATURE_DIMENSION,
        "sample_rate": config.TARGET_SAMPLE_RATE,
        "channels": config.TARGET_CHANNELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "n_mfcc": N_MFCC,
        "fmin": FMIN,
        "fmax": FMAX,
        "log_floor_db": LOG_FLOOR_DB,
        "layout": "mfcc_mean_std+delta_mean_std+spectral_scalars",
    }
