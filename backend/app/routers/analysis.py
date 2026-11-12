"""Cry analysis endpoint: upload/record -> decode -> features -> real model.

The endpoint never invents a result. When the trained ``.joblib`` file is not
present it returns ``model.connected = false`` together with the full audio
report, and no history record is created.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from .. import audio, config, db, features, model
from ..deps import api_error, device_token

router = APIRouter(prefix="/api", tags=["analysis"])


def _compact_result(payload: dict) -> dict:
    """Trim the report before storing it in SQLite (no heavy arrays)."""
    audio_report = payload["audio"]
    return {
        "prediction": payload.get("prediction"),
        "model": {
            "connected": payload["model"]["connected"],
            "version": payload["model"].get("version"),
        },
        "audio": {
            "duration_seconds": audio_report["duration_seconds"],
            "sample_rate": audio_report["sample_rate"],
            "quality": audio_report["quality"],
            "dominant_frequency_hz": audio_report["dominant_frequency_hz"],
            "frequency_bands": audio_report["frequency_bands"],
            "waveform": audio_report["waveform"],
        },
        "warnings": payload.get("warnings", []),
        "feature_vector": payload.get("feature_vector"),
    }


@router.get("/model/status")
def model_status() -> dict:
    status = model.registry.status()
    status["expected_path"] = str(config.MODEL_PATH)
    status["expected_meta_path"] = str(config.MODEL_META_PATH)
    status["meta_template"] = {
        "version": "wafaa-cry-1.0",
        "trained_at": "YYYY-MM-DD",
        "labels": config.CRY_CATEGORY_KEYS,
        "feature_vector": features.FEATURE_VECTOR_VERSION,
        "notes": "Describe the training data and preprocessing here.",
    }
    status["feature_pipeline"] = features.feature_metadata()
    return status


@router.post("/analysis")
async def analyse(
    file: UploadFile = File(..., description="Audio clip recorded or uploaded by the parent"),
    store_recording: bool = Form(default=False),
    baby_id: str | None = Form(default=None),
    device: str = Depends(device_token),
) -> dict:
    owner = db.owner_hash(device)

    if baby_id:
        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT id FROM babies WHERE id = ? AND owner = ?", (baby_id, owner)
            ).fetchone()
        if row is None:
            baby_id = None

    raw = await file.read()
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise api_error(
            413,
            "file_too_large",
            f"The file is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    try:
        decoded = audio.decode_audio(raw, file.filename)
    except audio.AudioError as exc:
        raise api_error(exc.status_code, exc.code, exc.message) from exc
    finally:
        await file.close()

    report = audio.analyse_signal(decoded)
    warnings: list[str] = []

    if report["quality"]["level"] == "poor":
        warnings.append("audio_quality_poor")
    elif report["quality"]["level"] == "fair":
        warnings.append("audio_quality_fair")

    vector = features.extract_features(decoded.samples, decoded.sample_rate)

    status = model.registry.status()
    prediction_payload: dict | None = None
    inference_error: dict | None = None

    if status["connected"]:
        try:
            prediction = model.registry.predict(vector)
            label = model.category_meta(prediction.category) if prediction.category else None
            prediction_payload = {
                "category": prediction.category,
                "raw_label": prediction.raw_label,
                "label_en": label["label_en"] if label else prediction.raw_label,
                "label_ar": label["label_ar"] if label else prediction.raw_label,
                "confidence": prediction.confidence,
                "confidence_available": prediction.confidence is not None,
                "probabilities": prediction.probabilities,
            }
            warnings.extend(prediction.warnings)
        except model.ModelNotConnected as exc:
            status = model.registry.status()
            inference_error = {"code": exc.code, "message": exc.message}
        except model.InferenceError as exc:
            inference_error = {
                "code": "inference_failed",
                "message": (
                    f"{exc} No prediction is shown, because a result that cannot be "
                    "trusted must not be displayed."
                ),
            }
        except Exception as exc:  # noqa: BLE001 - never fake a result
            inference_error = {
                "code": "inference_failed",
                "message": f"The model could not analyse this clip: {exc}",
            }

    response: dict = {
        "audio": report,
        "model": {
            "connected": status["connected"],
            "message_code": status["message_code"],
            "message": status["message"],
            "version": status["model_version"],
            "feature_vector": status["feature_vector"],
        },
        "prediction": prediction_payload,
        "error": inference_error,
        "warnings": warnings,
        "feature_vector": features.FEATURE_VECTOR_VERSION,
        "consent_store_recording": bool(store_recording),
        "recording_stored": False,
        "analysis_id": None,
        "saved": False,
        "disclaimer_code": "not_a_medical_device",
    }

    if prediction_payload is not None:
        analysis_id = db.new_id()
        recording_path: str | None = None
        if store_recording:
            target = config.RECORDINGS_DIR / f"{analysis_id}.wav"
            try:
                audio.write_pcm16_wav(decoded.samples, target, decoded.sample_rate)
                recording_path = str(target.relative_to(config.DATA_DIR))
                response["recording_stored"] = True
            except OSError:
                warnings.append("recording_store_failed")

        with db.get_conn() as conn:
            conn.execute(
                "INSERT INTO analyses (id, owner, baby_id, created_at, category, "
                "category_label, confidence, model_version, duration_seconds, "
                "sample_rate, quality_level, dominant_frequency_hz, result, "
                "recording_saved, recording_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?)",
                (
                    analysis_id,
                    owner,
                    baby_id,
                    db.utcnow(),
                    prediction_payload["category"],
                    prediction_payload["label_en"],
                    prediction_payload["confidence"],
                    status["model_version"],
                    report["duration_seconds"],
                    report["sample_rate"],
                    report["quality"]["level"],
                    report["dominant_frequency_hz"],
                    json.dumps(_compact_result(response), ensure_ascii=False),
                    int(response["recording_stored"]),
                    recording_path,
                ),
            )
        response["analysis_id"] = analysis_id
        response["saved"] = True

    return response


@router.get("/analysis/history")
def analysis_history(limit: int = 30, device: str = Depends(device_token)) -> dict:
    owner = db.owner_hash(device)
    limit = max(1, min(limit, 100))
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE owner = ? ORDER BY created_at DESC LIMIT ?",
            (owner, limit),
        ).fetchall()
    items = []
    for row in rows:
        try:
            result = json.loads(row["result"])
        except json.JSONDecodeError:
            result = {}
        items.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "baby_id": row["baby_id"],
                "category": row["category"],
                "category_label": row["category_label"],
                "confidence": row["confidence"],
                "model_version": row["model_version"],
                "duration_seconds": row["duration_seconds"],
                "sample_rate": row["sample_rate"],
                "quality_level": row["quality_level"],
                "dominant_frequency_hz": row["dominant_frequency_hz"],
                "recording_saved": bool(row["recording_saved"]),
                "result": result,
            }
        )
    return {"items": items, "count": len(items)}


@router.delete("/analysis/history/{analysis_id}")
def delete_analysis(analysis_id: str, device: str = Depends(device_token)) -> dict:
    owner = db.owner_hash(device)
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT recording_path FROM analyses WHERE id = ? AND owner = ?",
            (analysis_id, owner),
        ).fetchone()
        if row is None:
            raise api_error(404, "analysis_not_found", "This analysis result was not found.")
        conn.execute("DELETE FROM analyses WHERE id = ? AND owner = ?", (analysis_id, owner))
    _remove_recording(row["recording_path"])
    return {"deleted": True, "id": analysis_id}


@router.delete("/analysis/history")
def delete_all_analyses(device: str = Depends(device_token)) -> dict:
    owner = db.owner_hash(device)
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT recording_path FROM analyses WHERE owner = ?", (owner,)
        ).fetchall()
        conn.execute("DELETE FROM analyses WHERE owner = ?", (owner,))
    for row in rows:
        _remove_recording(row["recording_path"])
    return {"deleted": True, "count": len(rows)}


def _remove_recording(relative_path: str | None) -> None:
    if not relative_path:
        return
    candidate = (config.DATA_DIR / relative_path).resolve()
    try:
        if candidate.is_file() and config.DATA_DIR.resolve() in candidate.parents:
            candidate.unlink()
    except OSError:
        pass
