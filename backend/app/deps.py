"""Shared FastAPI dependencies: device scoping, validation and error helpers."""

from __future__ import annotations

import re

from fastapi import Header, HTTPException

_DEVICE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,80}$")


def device_token(x_device_id: str | None = Header(default=None, alias="X-Device-Id")) -> str:
    """Return the raw device token sent by the browser.

    The token is generated client-side once and stored in ``localStorage``. Only
    its hash is written to the database, and every query is scoped to it, so two
    devices never see each other's profiles or history.
    """
    if not x_device_id or not _DEVICE_PATTERN.match(x_device_id):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "device_id_required",
                "message": "A valid X-Device-Id header is required.",
            },
        )
    return x_device_id


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})
