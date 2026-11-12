"""Educational content and per-device settings."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends

from .. import config, db
from ..deps import api_error, device_token
from ..schemas import SettingsUpdate

router = APIRouter(prefix="/api", tags=["content"])

_CACHE: dict[str, object] = {"mtime": None, "data": None}


def load_articles() -> dict:
    path: Path = config.ARTICLES_PATH
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {"articles": [], "tips": [], "sources": []}
    if _CACHE["mtime"] != mtime or _CACHE["data"] is None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {"articles": [], "tips": [], "sources": []}
        _CACHE["mtime"] = mtime
        _CACHE["data"] = data
    return _CACHE["data"]  # type: ignore[return-value]


@router.get("/articles")
def list_articles() -> dict:
    data = load_articles()
    items = [
        {
            "slug": article["slug"],
            "title_ar": article["title_ar"],
            "title_en": article["title_en"],
            "summary_ar": article["summary_ar"],
            "summary_en": article["summary_en"],
            "read_minutes": article.get("read_minutes", 3),
            "icon": article.get("icon", "book"),
            "sources": article.get("sources", []),
        }
        for article in data.get("articles", [])
    ]
    return {
        "items": items,
        "urgent_warning_ar": data.get("urgent_warning_ar"),
        "urgent_warning_en": data.get("urgent_warning_en"),
        "sources": data.get("sources", []),
    }


@router.get("/articles/{slug}")
def get_article(slug: str) -> dict:
    data = load_articles()
    for article in data.get("articles", []):
        if article["slug"] == slug:
            return article
    raise api_error(404, "article_not_found", "This article was not found.")


@router.get("/tips")
def list_tips() -> dict:
    data = load_articles()
    return {
        "items": data.get("tips", []),
        "urgent_warning_ar": data.get("urgent_warning_ar"),
        "urgent_warning_en": data.get("urgent_warning_en"),
    }


@router.get("/settings")
def read_settings(device: str = Depends(device_token)) -> dict:
    return db.get_settings(db.owner_hash(device))


@router.put("/settings")
def write_settings(payload: SettingsUpdate, device: str = Depends(device_token)) -> dict:
    updates = payload.model_dump(exclude_none=True)
    return db.update_settings(db.owner_hash(device), updates)
