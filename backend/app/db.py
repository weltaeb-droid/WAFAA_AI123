"""SQLite persistence layer.

A lightweight, dependency-free SQLite database keeps baby profiles, analysis
history, favourites and preferences on the server.

Privacy model
-------------
There are no accounts. Every request carries a device token generated and kept
in the browser (`X-Device-Id`). Only a SHA-256 hash of that token is stored, and
every row is scoped to that hash, so one visitor's baby profile and analysis
history are never returned to another device.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS babies (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,
    date_of_birth TEXT NOT NULL,
    gender TEXT NOT NULL,
    weight_kg REAL,
    length_cm REAL,
    photo_data_url TEXT,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_babies_owner ON babies(owner);

CREATE TABLE IF NOT EXISTS profile_history (
    id TEXT PRIMARY KEY,
    baby_id TEXT NOT NULL,
    owner TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    changed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_profile_history_baby ON profile_history(baby_id);

CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    baby_id TEXT,
    created_at TEXT NOT NULL,
    category TEXT,
    category_label TEXT,
    confidence REAL,
    model_version TEXT,
    duration_seconds REAL NOT NULL,
    sample_rate INTEGER NOT NULL,
    quality_level TEXT NOT NULL,
    dominant_frequency_hz REAL,
    result TEXT NOT NULL,
    recording_saved INTEGER NOT NULL DEFAULT 0,
    recording_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_owner ON analyses(owner, created_at DESC);

CREATE TABLE IF NOT EXISTS favorites (
    owner TEXT NOT NULL,
    sound_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (owner, sound_id)
);

CREATE TABLE IF NOT EXISTS plays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner TEXT NOT NULL,
    sound_id TEXT NOT NULL,
    played_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plays_owner ON plays(owner, played_at DESC);

CREATE TABLE IF NOT EXISTS settings (
    owner TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (owner, key)
);
"""

DEFAULT_SETTINGS: dict[str, Any] = {
    "language": "ar",
    "store_recordings": False,
    "volume": 0.5,
    "tips_enabled": True,
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex


def owner_hash(device_id: str) -> str:
    """Hash the browser device token before it ever touches the database."""
    return hashlib.sha256(f"wafaa-ai::{device_id}".encode("utf-8")).hexdigest()


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    config.ensure_directories()
    with get_conn() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


# --- settings ----------------------------------------------------------------


def get_settings(owner: str) -> dict[str, Any]:
    values = dict(DEFAULT_SETTINGS)
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT key, value FROM settings WHERE owner = ?", (owner,)
        ).fetchall()
    for row in rows:
        try:
            values[row["key"]] = json.loads(row["value"])
        except json.JSONDecodeError:
            values[row["key"]] = row["value"]
    return values


def update_settings(owner: str, updates: dict[str, Any]) -> dict[str, Any]:
    allowed = set(DEFAULT_SETTINGS)
    with get_conn() as conn:
        for key, value in updates.items():
            if key not in allowed or value is None:
                continue
            conn.execute(
                "INSERT INTO settings (owner, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT(owner, key) DO UPDATE SET value = excluded.value",
                (owner, key, json.dumps(value)),
            )
    return get_settings(owner)
