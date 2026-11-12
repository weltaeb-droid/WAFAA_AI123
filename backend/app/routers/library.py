"""Soothing sound library: catalogue, favourites and recently played sounds."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends

from .. import config, db
from ..deps import api_error, device_token
from ..schemas import FavoritePayload

router = APIRouter(prefix="/api", tags=["library"])

_CACHE: dict[str, object] = {"mtime": None, "data": None}


def load_catalog() -> dict:
    path: Path = config.SOUNDS_MANIFEST
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {"categories": [], "items": [], "license_note": "", "safety_note": ""}
    if _CACHE["mtime"] != mtime or _CACHE["data"] is None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {"categories": [], "items": []}
        _CACHE["mtime"] = mtime
        _CACHE["data"] = data
    return _CACHE["data"]  # type: ignore[return-value]


def _favorite_ids(owner: str) -> set[str]:
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT sound_id FROM favorites WHERE owner = ?", (owner,)
        ).fetchall()
    return {row["sound_id"] for row in rows}


def _decorate(items: list[dict], favorites: set[str]) -> list[dict]:
    return [{**item, "is_favorite": item["id"] in favorites} for item in items]


@router.get("/sounds")
def list_sounds(device: str = Depends(device_token)) -> dict:
    catalog = load_catalog()
    favorites = _favorite_ids(db.owner_hash(device))
    items = _decorate(catalog.get("items", []), favorites)
    return {
        "items": items,
        "categories": catalog.get("categories", []),
        "license_note": catalog.get("license_note", ""),
        "safety_note": catalog.get("safety_note", ""),
        "attribution_note": catalog.get("attribution_note", ""),
        "favorites": sorted(favorites),
    }


@router.post("/sounds/{sound_id}/favorite")
def toggle_favorite(
    sound_id: str, payload: FavoritePayload, device: str = Depends(device_token)
) -> dict:
    catalog = load_catalog()
    known = {item["id"] for item in catalog.get("items", [])}
    if sound_id not in known:
        raise api_error(404, "sound_not_found", "This sound is not part of the library.")
    owner = db.owner_hash(device)
    with db.get_conn() as conn:
        if payload.favorite:
            conn.execute(
                "INSERT OR REPLACE INTO favorites (owner, sound_id, created_at) "
                "VALUES (?, ?, ?)",
                (owner, sound_id, db.utcnow()),
            )
        else:
            conn.execute(
                "DELETE FROM favorites WHERE owner = ? AND sound_id = ?", (owner, sound_id)
            )
    return {"sound_id": sound_id, "is_favorite": payload.favorite}


@router.get("/favorites")
def list_favorites(device: str = Depends(device_token)) -> dict:
    catalog = load_catalog()
    favorites = _favorite_ids(db.owner_hash(device))
    items = [item for item in catalog.get("items", []) if item["id"] in favorites]
    return {"items": _decorate(items, favorites), "count": len(items)}


@router.post("/sounds/{sound_id}/play")
def record_play(sound_id: str, device: str = Depends(device_token)) -> dict:
    catalog = load_catalog()
    known = {item["id"] for item in catalog.get("items", [])}
    if sound_id not in known:
        raise api_error(404, "sound_not_found", "This sound is not part of the library.")
    owner = db.owner_hash(device)
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO plays (owner, sound_id, played_at) VALUES (?, ?, ?)",
            (owner, sound_id, db.utcnow()),
        )
        # Keep only the most recent plays per device.
        conn.execute(
            "DELETE FROM plays WHERE owner = ? AND id NOT IN ("
            "SELECT id FROM plays WHERE owner = ? ORDER BY played_at DESC LIMIT 200)",
            (owner, owner),
        )
    return {"recorded": True}


@router.get("/sounds/recent")
def recent_sounds(device: str = Depends(device_token)) -> dict:
    catalog = load_catalog()
    by_id = {item["id"]: item for item in catalog.get("items", [])}
    favorites = _favorite_ids(db.owner_hash(device))
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT sound_id, MAX(played_at) AS played_at FROM plays WHERE owner = ? "
            "GROUP BY sound_id ORDER BY played_at DESC LIMIT 6",
            (db.owner_hash(device),),
        ).fetchall()
    items = []
    for row in rows:
        item = by_id.get(row["sound_id"])
        if item:
            items.append({**item, "is_favorite": item["id"] in favorites,
                          "played_at": row["played_at"]})
    return {"items": items}
