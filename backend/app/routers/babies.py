"""Baby profile CRUD, age calculation and profile history."""

from __future__ import annotations

import json
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException

from .. import config, db
from ..deps import api_error, device_token
from ..schemas import BabyCreate, BabyUpdate

router = APIRouter(prefix="/api/babies", tags=["babies"])


def _age_from_dob(dob_text: str) -> dict:
    """Age in days/weeks/months. Informational only — not a growth assessment."""
    try:
        dob = date.fromisoformat(dob_text)
    except ValueError:
        return {"days": 0, "weeks": 0, "months": 0}
    today = date.today()
    days = max(0, (today - dob).days)
    months = (today.year - dob.year) * 12 + (today.month - dob.month)
    if today.day < dob.day:
        months -= 1
    return {"days": days, "weeks": days // 7, "months": max(0, months)}


def _serialise(row) -> dict:
    item = dict(row)
    item["age"] = _age_from_dob(item["date_of_birth"])
    return item


def _snapshot(item: dict) -> dict:
    return {
        "name": item["name"],
        "date_of_birth": item["date_of_birth"],
        "gender": item["gender"],
        "weight_kg": item["weight_kg"],
        "length_cm": item["length_cm"],
        "notes": item["notes"],
        "recorded_at": item["updated_at"],
    }


def _fetch_baby(conn, owner: str, baby_id: str) -> dict:
    row = conn.execute(
        "SELECT * FROM babies WHERE id = ? AND owner = ?", (baby_id, owner)
    ).fetchone()
    if row is None:
        raise api_error(404, "baby_not_found", "This baby profile was not found.")
    return dict(row)


@router.get("")
def list_babies(device: str = Depends(device_token)) -> dict:
    owner = db.owner_hash(device)
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM babies WHERE owner = ? ORDER BY created_at ASC", (owner,)
        ).fetchall()
    return {"items": [_serialise(row) for row in rows]}


@router.post("", status_code=201)
def create_baby(payload: BabyCreate, device: str = Depends(device_token)) -> dict:
    owner = db.owner_hash(device)
    baby_id = db.new_id()
    now = db.utcnow()
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO babies (id, owner, name, date_of_birth, gender, weight_kg, "
            "length_cm, photo_data_url, notes, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                baby_id,
                owner,
                payload.name,
                payload.date_of_birth.isoformat(),
                payload.gender,
                payload.weight_kg,
                payload.length_cm,
                payload.photo_data_url,
                payload.notes,
                now,
                now,
            ),
        )
        row = conn.execute("SELECT * FROM babies WHERE id = ?", (baby_id,)).fetchone()
        created = dict(row)
        conn.execute(
            "INSERT INTO profile_history (id, baby_id, owner, snapshot, changed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                db.new_id(),
                baby_id,
                owner,
                json.dumps(_snapshot(created), ensure_ascii=False),
                now,
            ),
        )
    return _serialise(created)


@router.get("/{baby_id}")
def get_baby(baby_id: str, device: str = Depends(device_token)) -> dict:
    owner = db.owner_hash(device)
    with db.get_conn() as conn:
        baby = _fetch_baby(conn, owner, baby_id)
        history = conn.execute(
            "SELECT snapshot, changed_at FROM profile_history WHERE baby_id = ? "
            "AND owner = ? ORDER BY changed_at DESC LIMIT 50",
            (baby_id, owner),
        ).fetchall()
    result = _serialise(baby)
    result["history"] = [
        {**json.loads(row["snapshot"]), "changed_at": row["changed_at"]} for row in history
    ]
    return result


@router.put("/{baby_id}")
def update_baby(baby_id: str, payload: BabyUpdate, device: str = Depends(device_token)) -> dict:
    owner = db.owner_hash(device)
    now = db.utcnow()
    with db.get_conn() as conn:
        current = _fetch_baby(conn, owner, baby_id)
        conn.execute(
            "INSERT INTO profile_history (id, baby_id, owner, snapshot, changed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                db.new_id(),
                baby_id,
                owner,
                json.dumps(_snapshot(current), ensure_ascii=False),
                now,
            ),
        )
        conn.execute(
            "UPDATE babies SET name = ?, date_of_birth = ?, gender = ?, weight_kg = ?, "
            "length_cm = ?, photo_data_url = ?, notes = ?, updated_at = ? "
            "WHERE id = ? AND owner = ?",
            (
                payload.name,
                payload.date_of_birth.isoformat(),
                payload.gender,
                payload.weight_kg,
                payload.length_cm,
                payload.photo_data_url,
                payload.notes,
                now,
                baby_id,
                owner,
            ),
        )
        row = conn.execute("SELECT * FROM babies WHERE id = ?", (baby_id,)).fetchone()
    return _serialise(dict(row))


@router.delete("/{baby_id}", status_code=200)
def delete_baby(baby_id: str, device: str = Depends(device_token)) -> dict:
    owner = db.owner_hash(device)
    with db.get_conn() as conn:
        _fetch_baby(conn, owner, baby_id)
        conn.execute("DELETE FROM profile_history WHERE baby_id = ? AND owner = ?", (baby_id, owner))
        conn.execute("DELETE FROM babies WHERE id = ? AND owner = ?", (baby_id, owner))
        conn.execute(
            "UPDATE analyses SET baby_id = NULL WHERE baby_id = ? AND owner = ?",
            (baby_id, owner),
        )
    return {"deleted": True, "id": baby_id}


@router.get("/{baby_id}/history")
def baby_history(baby_id: str, device: str = Depends(device_token)) -> dict:
    owner = db.owner_hash(device)
    with db.get_conn() as conn:
        _fetch_baby(conn, owner, baby_id)
        rows = conn.execute(
            "SELECT snapshot, changed_at FROM profile_history WHERE baby_id = ? "
            "AND owner = ? ORDER BY changed_at DESC LIMIT 100",
            (baby_id, owner),
        ).fetchall()
    return {
        "items": [
            {**json.loads(row["snapshot"]), "changed_at": row["changed_at"]} for row in rows
        ],
        "note": (
            "Measurements recorded by the parent. WAFAA AI does not classify growth "
            "as healthy or unhealthy; use age- and sex-specific WHO growth charts "
            "with a paediatric professional."
        ),
    }
