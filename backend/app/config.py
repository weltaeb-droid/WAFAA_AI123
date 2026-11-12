"""Central configuration for the WAFAA AI backend.

Every path can be overridden with an environment variable so the API can run
from a container, from a local checkout, or from a custom deployment folder.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
REPO_DIR = BACKEND_DIR.parent


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(default)


# --- Storage -----------------------------------------------------------------
DATA_DIR = _path_from_env("WAFAA_DATA_DIR", BACKEND_DIR / "data")
RECORDINGS_DIR = DATA_DIR / "recordings"
DB_PATH = _path_from_env("WAFAA_DB_PATH", DATA_DIR / "wafaa.sqlite3")

# --- Model -------------------------------------------------------------------
MODEL_DIR = _path_from_env("WAFAA_MODEL_DIR", BACKEND_DIR / "models")
MODEL_PATH = _path_from_env("WAFAA_MODEL_PATH", MODEL_DIR / "wafaa_cry_model.joblib")
MODEL_META_PATH = _path_from_env(
    "WAFAA_MODEL_META_PATH", MODEL_DIR / "wafaa_cry_model.meta.json"
)

# --- Content / static --------------------------------------------------------
STATIC_AUDIO_DIR = APP_DIR / "static" / "audio"
SOUNDS_MANIFEST = APP_DIR / "data" / "sounds.json"
ARTICLES_PATH = APP_DIR / "data" / "articles.json"
FRONTEND_DIST = _path_from_env("WAFAA_FRONTEND_DIST", REPO_DIR / "frontend" / "dist")

# --- Audio contract ----------------------------------------------------------
# The trained WAFAA AI model expects 16 kHz mono audio. Every uploaded or
# recorded clip is decoded, downmixed to mono and resampled to this rate before
# feature extraction, so the API always feeds the model identical input.
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_AUDIO_SECONDS = 60.0
MIN_AUDIO_SECONDS = 1.0
MAX_RECORDING_BYTES = 6 * 1024 * 1024  # stored consented clips are trimmed copies
ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".aac",
    ".ogg",
    ".oga",
    ".opus",
    ".webm",
    ".flac",
    ".amr",
    ".3gp",
    ".caf",
    ".aiff",
}

# --- Cry categories ----------------------------------------------------------
# The five classes the WAFAA AI model is trained on.
CRY_CATEGORIES: list[dict[str, str]] = [
    {"key": "belly_pain", "label_en": "Belly pain", "label_ar": "ألم في البطن"},
    {"key": "burping", "label_en": "Burping", "label_ar": "بحاجة إلى التجشؤ"},
    {"key": "discomfort", "label_en": "Discomfort", "label_ar": "انزعاج"},
    {"key": "hungry", "label_en": "Hungry", "label_ar": "جوع"},
    {"key": "tired", "label_en": "Tired", "label_ar": "تعب / نعاس"},
]
CRY_CATEGORY_KEYS = [item["key"] for item in CRY_CATEGORIES]

# Limits for user supplied content.
MAX_NAME_LENGTH = 60
MAX_NOTES_LENGTH = 2000
MAX_PHOTO_DATA_URL_BYTES = 700_000  # ~500 KB binary once base64 decoded

APP_NAME = "WAFAA AI"
APP_VERSION = "1.0.0"


def ensure_directories() -> None:
    """Create the writable folders the API needs (safe to call repeatedly)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
