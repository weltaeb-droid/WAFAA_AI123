"""Request/response validation models (Pydantic v2)."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from . import config

Gender = Literal["girl", "boy", "unspecified"]


class BabyBase(BaseModel):
    name: str = Field(min_length=1, max_length=config.MAX_NAME_LENGTH)
    date_of_birth: date
    gender: Gender = "unspecified"
    weight_kg: float | None = Field(default=None, ge=0.4, le=45)
    length_cm: float | None = Field(default=None, ge=20, le=150)
    photo_data_url: str | None = None
    notes: str = Field(default="", max_length=config.MAX_NOTES_LENGTH)

    @field_validator("name", "notes")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @field_validator("date_of_birth")
    @classmethod
    def _not_in_future(cls, value: date) -> date:
        today = date.today()
        if value > today:
            raise ValueError("date_of_birth cannot be in the future")
        if (today - value).days > 365 * 6:
            raise ValueError("date_of_birth is too far in the past for an infant profile")
        return value

    @field_validator("photo_data_url")
    @classmethod
    def _photo(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if not value.startswith("data:image/"):
            raise ValueError("photo must be a data:image/... URL")
        if len(value) > config.MAX_PHOTO_DATA_URL_BYTES:
            raise ValueError("photo is too large; please use an image under 500 KB")
        return value


class BabyCreate(BabyBase):
    pass


class BabyUpdate(BabyBase):
    pass


class FavoritePayload(BaseModel):
    favorite: bool = True


class SettingsUpdate(BaseModel):
    language: Literal["ar", "en"] | None = None
    store_recordings: bool | None = None
    volume: float | None = Field(default=None, ge=0.0, le=1.0)
    tips_enabled: bool | None = None
