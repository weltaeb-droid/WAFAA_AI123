"""Generate the original WAFAA AI soothing-sound library.

Every clip is synthesized from scratch by this script (no samples, no third
party recordings), so the whole library can be released as original content.
Output:

* ``backend/app/static/audio/<id>.mp3``  — mono, 22.05 kHz, 64 kbps
* ``backend/app/data/sounds.json``        — catalogue consumed by the API

Run from the repository root:

    python3 scripts/generate_audio.py
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22_050
DURATION = 40.0
TARGET_PEAK = 0.5  # ≈ -6 dBFS: intentionally gentle output level
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "backend" / "app" / "static" / "audio"
MANIFEST = ROOT / "backend" / "app" / "data" / "sounds.json"

RNG = np.random.default_rng(20240607)


def t_axis(duration: float = DURATION) -> np.ndarray:
    return np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE


def smooth_noise(duration: float, cutoff_hz: float, seed: int) -> np.ndarray:
    """One-pole low-pass filtered white noise."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(int(SAMPLE_RATE * duration))
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / SAMPLE_RATE)
    out = np.zeros_like(white)
    acc = 0.0
    for index in range(len(white)):
        acc += alpha * (white[index] - acc)
        out[index] = acc
    return out


def pink_noise(duration: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(int(SAMPLE_RATE * duration))
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(white), d=1.0 / SAMPLE_RATE)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    shaped = np.fft.irfft(spectrum * scale, n=len(white))
    return shaped


def slow_lfo(t: np.ndarray, rate_hz: float, phase: float = 0.0) -> np.ndarray:
    return 0.5 + 0.5 * np.sin(2.0 * math.pi * rate_hz * t + phase)


def fade(signal: np.ndarray, seconds: float = 1.5) -> np.ndarray:
    length = min(len(signal), int(seconds * SAMPLE_RATE))
    if length <= 1:
        return signal
    ramp = np.linspace(0.0, 1.0, length)
    signal[:length] *= ramp
    signal[-length:] *= ramp[::-1]
    return signal


def normalise(signal: np.ndarray, peak: float = TARGET_PEAK) -> np.ndarray:
    maximum = float(np.max(np.abs(signal))) or 1.0
    return signal * (peak / maximum)


def womb_rumble() -> np.ndarray:
    t = t_axis()
    brown = smooth_noise(DURATION, 55.0, 11)
    brown = np.cumsum(brown)
    brown -= np.linspace(brown[0], brown[-1], len(brown))
    brown = normalise(brown, 0.6)
    drone = (
        0.55 * np.sin(2.0 * math.pi * 84.0 * t)
        + 0.28 * np.sin(2.0 * math.pi * 126.0 * t + 0.7)
        + 0.14 * np.sin(2.0 * math.pi * 168.0 * t + 1.9)
    )
    drone *= 0.55 + 0.45 * slow_lfo(t, 0.07)
    breath = smooth_noise(DURATION, 420.0, 12) * 0.35
    breath *= 0.6 + 0.4 * slow_lfo(t, 0.13, 1.2)
    return normalise(0.75 * drone + 0.5 * brown + 0.3 * breath)


def womb_hush() -> np.ndarray:
    t = t_axis()
    hiss = smooth_noise(DURATION, 900.0, 21)
    hiss = normalise(hiss, 1.0)
    swell = 0.35 + 0.65 * slow_lfo(t, 0.11)
    low = np.sin(2.0 * math.pi * 110.0 * t) * 0.18
    return normalise(hiss * swell + low)


def heartbeat() -> np.ndarray:
    t = t_axis()
    bpm = 62.0
    period = 60.0 / bpm
    out = np.zeros_like(t)
    thump_len = int(0.22 * SAMPLE_RATE)
    thump_t = np.arange(thump_len) / SAMPLE_RATE
    lub = np.sin(2.0 * math.pi * 52.0 * thump_t) * np.exp(-thump_t * 22.0)
    dub = np.sin(2.0 * math.pi * 64.0 * thump_t) * np.exp(-thump_t * 26.0)
    start = 0.0
    while start < DURATION - period:
        for offset, shape, gain in ((0.0, lub, 1.0), (0.30, dub, 0.62)):
            index = int((start + offset) * SAMPLE_RATE)
            end = min(index + thump_len, len(out))
            if index < len(out):
                out[index:end] += shape[: end - index] * gain
        start += period
    bed = smooth_noise(DURATION, 200.0, 31) * 0.22
    return normalise(out + bed)


def soft_white_noise() -> np.ndarray:
    t = t_axis()
    noise = normalise(smooth_noise(DURATION, 2600.0, 41), 1.0)
    pulse = 0.55 + 0.45 * slow_lfo(t, 0.55)  # gentle rhythmic swell, not a beat
    return normalise(noise * pulse, 0.42)


def pink_drift() -> np.ndarray:
    t = t_axis()
    noise = normalise(pink_noise(DURATION, 51), 1.0)
    drift = 0.6 + 0.4 * slow_lfo(t, 0.045, 2.1)
    return normalise(noise * drift, 0.4)


def water_stream() -> np.ndarray:
    t = t_axis()
    base = smooth_noise(DURATION, 1500.0, 61)
    base = normalise(base, 1.0)
    swirl = 0.45 + 0.55 * slow_lfo(t, 0.17, 0.4) * slow_lfo(t, 0.23, 1.1)
    bubbles = np.zeros_like(t)
    rng = np.random.default_rng(71)
    for _ in range(26):
        start = float(rng.uniform(0.5, DURATION - 1.5))
        length = float(rng.uniform(0.08, 0.22))
        freq = float(rng.uniform(320.0, 900.0))
        idx = int(start * SAMPLE_RATE)
        count = int(length * SAMPLE_RATE)
        if idx + count >= len(t):
            continue
        local_t = np.arange(count) / SAMPLE_RATE
        env = np.sin(np.pi * local_t / length) ** 2
        bubbles[idx : idx + count] += np.sin(
            2.0 * math.pi * (freq + 240.0 * local_t / length) * local_t
        ) * env * 0.22
    return normalise(base * swirl * 0.8 + bubbles)


def water_drops() -> np.ndarray:
    t = t_axis()
    bed = normalise(smooth_noise(DURATION, 700.0, 81), 1.0) * 0.35
    rng = np.random.default_rng(82)
    drops = np.zeros_like(t)
    for _ in range(16):
        start = float(rng.uniform(0.8, DURATION - 1.2))
        freq = float(rng.uniform(600.0, 1200.0))
        idx = int(start * SAMPLE_RATE)
        count = int(0.5 * SAMPLE_RATE)
        if idx + count >= len(t):
            continue
        local_t = np.arange(count) / SAMPLE_RATE
        env = np.exp(-local_t * 9.0)
        drops[idx : idx + count] += np.sin(2.0 * math.pi * freq * local_t) * env * 0.3
    return normalise(bed + drops)


def ambient_pad(root_hz: float, seed: int) -> np.ndarray:
    t = t_axis()
    partials = [1.0, 1.5, 2.0, 2.5, 3.0]
    gains = [0.5, 0.28, 0.2, 0.1, 0.06]
    out = np.zeros_like(t)
    rng = np.random.default_rng(seed)
    for index, (ratio, gain) in enumerate(zip(partials, gains)):
        detune = 1.0 + float(rng.uniform(-0.0018, 0.0018))
        lfo_rate = 0.05 + 0.03 * index
        amp = 0.7 + 0.3 * slow_lfo(t, lfo_rate, float(rng.uniform(0, 6.28)))
        out += gain * amp * np.sin(2.0 * math.pi * root_hz * ratio * detune * t)
    air = normalise(smooth_noise(DURATION, 500.0, seed + 5), 1.0) * 0.12
    return normalise(out * 0.9 + air)


def lullaby(root_hz: float, seed: int, pattern: list[int]) -> np.ndarray:
    """A slow, original pentatonic melody with soft bell-like tones."""
    t = t_axis()
    out = np.zeros_like(t)
    scale = [0, 2, 4, 7, 9, 12]
    beat = 0.9
    position = 1.0
    rng = np.random.default_rng(seed)
    note_index = 0
    while position < DURATION - 2.0:
        step = pattern[note_index % len(pattern)]
        semitones = scale[step % len(scale)] + 12 * (step // len(scale))
        freq = root_hz * (2.0 ** (semitones / 12.0))
        length = beat * float(rng.choice([1.0, 1.0, 1.5, 2.0]))
        count = int(length * SAMPLE_RATE)
        idx = int(position * SAMPLE_RATE)
        if idx + count >= len(t):
            break
        local_t = np.arange(count) / SAMPLE_RATE
        env = np.exp(-local_t * 1.9) * (1.0 - np.exp(-local_t * 25.0))
        tone = (
            np.sin(2.0 * math.pi * freq * local_t)
            + 0.25 * np.sin(2.0 * math.pi * freq * 2.0 * local_t)
            + 0.08 * np.sin(2.0 * math.pi * freq * 3.0 * local_t)
        )
        out[idx : idx + count] += tone * env * 0.5
        position += length
        note_index += 1
    pad = ambient_pad(root_hz / 2.0, seed + 3) * 0.35
    return normalise(out + pad, 0.45)


def render(signal: np.ndarray, path_stem: Path) -> None:
    signal = fade(np.clip(signal, -1.0, 1.0))
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "clip.wav"
        pcm = (signal * 32767.0).astype("<i2")
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(pcm.tobytes())
        mp3_path = path_stem.with_suffix(".mp3")
        if shutil.which("ffmpeg"):
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(wav_path), "-ac", "1", "-ar", str(SAMPLE_RATE),
                    "-b:a", "64k", "-codec:a", "libmp3lame", str(mp3_path),
                ],
                check=True,
            )
        else:  # pragma: no cover - fallback when ffmpeg is unavailable
            shutil.copyfile(wav_path, path_stem.with_suffix(".wav"))


SOUNDS = [
    {
        "id": "womb-rumble",
        "category": "womb",
        "icon": "womb",
        "title_ar": "همهمة الرحم الهادئة",
        "title_en": "Calm womb-inspired hum",
        "description_ar": "مزيج منخفض من نغمات عميقة وضجيج ناعم، مستوحى من أوصاف الأصوات المنخفضة التي قد يسمعها الجنين. ليس تسجيلًا من داخل الرحم.",
        "description_en": "A low blend of deep tones and soft noise, inspired by descriptions of the muffled low sounds a fetus may hear. Not a recording from inside the uterus.",
        "generator": lambda: womb_rumble(),
    },
    {
        "id": "womb-hush",
        "category": "womb",
        "icon": "womb",
        "title_ar": "همس دافئ",
        "title_en": "Warm hush",
        "description_ar": "ضجيج ناعم متدرّج بنبض بطيء جدًا، يشبه صوت \"شششش\" الهادئ. مستوحى من الأصوات المحيطة وليس تسجيلًا حقيقيًا.",
        "description_en": "Soft swelling noise with a very slow pulse, similar to a gentle \"shhh\" sound. Inspired by ambient sound, not a real recording.",
        "generator": lambda: womb_hush(),
    },
    {
        "id": "heartbeat-gentle",
        "category": "heartbeat",
        "icon": "heart",
        "title_ar": "نبض قلب هادئ",
        "title_en": "Gentle heartbeat",
        "description_ar": "إيقاع منتظم بسرعة 62 نبضة في الدقيقة تقريبًا مع طبقة صوتية دافئة. إيقاع مُولَّد ومستوحى من نبض القلب.",
        "description_en": "A regular rhythm at roughly 62 beats per minute with a warm bed of sound. A generated heartbeat-inspired rhythm.",
        "generator": lambda: heartbeat(),
    },
    {
        "id": "white-noise-soft",
        "category": "white_noise",
        "icon": "waves",
        "title_ar": "ضجيج أبيض ناعم",
        "title_en": "Soft white noise",
        "description_ar": "ضجيج أبيض مُرشَّح بنبض خفيف جدًا في مستوى الصوت. أبقِ الصوت منخفضًا.",
        "description_en": "Filtered white noise with a very gentle volume swell. Keep the volume low.",
        "generator": lambda: soft_white_noise(),
    },
    {
        "id": "pink-noise-drift",
        "category": "white_noise",
        "icon": "waves",
        "title_ar": "ضجيج وردي متحرك",
        "title_en": "Pink noise drift",
        "description_ar": "ضجيج وردي بطيف أكثر نعومة من الأبيض، يتحرك ببطء شديد.",
        "description_en": "Pink noise with a softer spectrum than white noise, drifting very slowly.",
        "generator": lambda: pink_drift(),
    },
    {
        "id": "water-stream",
        "category": "water",
        "icon": "droplet",
        "title_ar": "جدول ماء هادئ",
        "title_en": "Quiet water stream",
        "description_ar": "ماء جارٍ مُولَّد رقميًا مع فقاعات لطيفة متفرقة.",
        "description_en": "Digitally generated flowing water with gentle, occasional bubbles.",
        "generator": lambda: water_stream(),
    },
    {
        "id": "water-drops",
        "category": "water",
        "icon": "droplet",
        "title_ar": "قطرات ماء",
        "title_en": "Water drops",
        "description_ar": "قطرات متباعدة فوق خلفية مائية ناعمة.",
        "description_en": "Sparse droplets over a soft watery background.",
        "generator": lambda: water_drops(),
    },
    {
        "id": "ambient-lavender",
        "category": "ambient",
        "icon": "sparkle",
        "title_ar": "محيط اللافندر",
        "title_en": "Lavender ambience",
        "description_ar": "طبقات نغمية بطيئة بدون كلمات، من تأليف وتوليد WAFAA AI.",
        "description_en": "Slow tonal layers without vocals, composed and generated for WAFAA AI.",
        "generator": lambda: ambient_pad(110.0, 91),
    },
    {
        "id": "ambient-moonlight",
        "category": "ambient",
        "icon": "moon",
        "title_ar": "ضوء القمر",
        "title_en": "Moonlight ambience",
        "description_ar": "محيط هادئ بنغمات أعمق وحركة بطيئة جدًا.",
        "description_en": "A calm ambient piece with deeper tones and very slow movement.",
        "generator": lambda: ambient_pad(82.5, 92),
    },
    {
        "id": "lullaby-stars",
        "category": "lullaby",
        "icon": "star",
        "title_ar": "تهويدة النجوم",
        "title_en": "Stars lullaby",
        "description_ar": "لحن أصلي بطيء بنغمات ناعمة، بدون كلمات.",
        "description_en": "An original slow melody with soft tones, no words.",
        "generator": lambda: lullaby(196.0, 101, [0, 2, 4, 3, 1, 2, 0, 4]),
    },
    {
        "id": "lullaby-dream",
        "category": "lullaby",
        "icon": "star",
        "title_ar": "تهويدة الحلم",
        "title_en": "Dream lullaby",
        "description_ar": "لحن أصلي آخر بنغمات أعمق وحركة أبطأ.",
        "description_en": "Another original melody with deeper tones and a slower pace.",
        "generator": lambda: lullaby(147.0, 102, [1, 3, 5, 4, 2, 0, 3, 1]),
    },
]

CATEGORIES = [
    {"key": "womb", "title_ar": "مستوحى من الرحم", "title_en": "Womb-inspired"},
    {"key": "heartbeat", "title_ar": "نبض القلب", "title_en": "Heartbeat-inspired"},
    {"key": "white_noise", "title_ar": "ضجيج ناعم", "title_en": "Soft noise"},
    {"key": "water", "title_ar": "أصوات الماء", "title_en": "Water"},
    {"key": "ambient", "title_ar": "محيط هادئ", "title_en": "Ambient"},
    {"key": "lullaby", "title_ar": "تهويدات", "title_en": "Lullabies"},
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for spec in SOUNDS:
        signal = spec["generator"]()
        path_stem = OUTPUT_DIR / spec["id"]
        render(signal, path_stem)
        suffix = ".mp3" if (path_stem.with_suffix(".mp3")).exists() else ".wav"
        items.append(
            {
                "id": spec["id"],
                "category": spec["category"],
                "icon": spec["icon"],
                "title_ar": spec["title_ar"],
                "title_en": spec["title_en"],
                "description_ar": spec["description_ar"],
                "description_en": spec["description_en"],
                "duration_seconds": round(len(signal) / SAMPLE_RATE, 1),
                "audio_url": f"/media/audio/{spec['id']}{suffix}",
                "license": "Original synthesized audio, generated for WAFAA AI (CC0 dedication)",
                "source_kind": "synthesized",
            }
        )
        print(f"rendered {spec['id']}")

    manifest = {
        "categories": CATEGORIES,
        "items": items,
        "license_note": (
            "Every clip is original audio synthesized by scripts/generate_audio.py for "
            "WAFAA AI and dedicated to the public domain (CC0). No third-party "
            "recordings are used."
        ),
        "attribution_note": (
            "These are original soundscapes inspired by descriptions of the prenatal "
            "auditory environment. They are not recordings made inside the uterus and "
            "do not reproduce the womb exactly."
        ),
        "safety_note": (
            "Keep playback at a low volume, never place headphones or a speaker "
            "directly against your baby's ears, and never leave audio playing loudly "
            "near the baby."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST} with {len(items)} items")


if __name__ == "__main__":
    main()
