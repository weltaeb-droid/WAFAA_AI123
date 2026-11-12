"""Secure loader for the trained WAFAA AI cry-classification model.

Design rules
------------
* The model is loaded **only** from a server-side path configured in
  ``config.py`` (``backend/models/wafaa_cry_model.joblib`` by default). Nothing
  uploaded by a browser is ever deserialised.
* If the file is absent, the API reports ``connected: false`` with a clear
  message. It never fabricates a prediction, a label or a confidence value.
* Confidence is returned **only** when the estimator exposes a valid
  probability output (``predict_proba``) whose values look like real
  probabilities. Otherwise ``confidence`` is ``None`` and the UI hides it.
* An optional sidecar ``wafaa_cry_model.meta.json`` declares the class order,
  feature-vector version and human readable version string. It is also the
  place to record which preprocessing the model expects.

Security note: ``joblib``/``pickle`` files can execute code when loaded. Only
place model files you produced or trust in ``backend/models/``; the API pins the
file name and, optionally, a SHA-256 digest (``WAFAA_MODEL_SHA256``).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import config, features

_MODEL_LOCK = threading.Lock()

# Normalised label aliases -> canonical WAFAA AI category.
_LABEL_ALIASES: dict[str, set[str]] = {
    "belly_pain": {
        "belly_pain",
        "bellypain",
        "belly_ache",
        "pain",
        "colic",
        "stomach_pain",
        "tummy_pain",
        "abdomen_pain",
        "مغص",
        "ألم",
        "الم",
        "ألم_البطن",
        "وجع",
    },
    "burping": {
        "burping",
        "burp",
        "needs_burp",
        "gas",
        "wind",
        "تجشؤ",
        "تجشو",
        "غازات",
        "بحاجة_إلى_التجشؤ",
    },
    "discomfort": {
        "discomfort",
        "uncomfortable",
        "irritated",
        "diaper",
        "wet",
        "انزعاج",
        "عدم_راحة",
        "غير_مرتاح",
    },
    "hungry": {
        "hungry",
        "hunger",
        "feeding",
        "feed",
        "hungry_cry",
        "جوع",
        "جائع",
        "جوعان",
    },
    "tired": {
        "tired",
        "sleepy",
        "sleep",
        "fatigue",
        "تعب",
        "نعاس",
        "متعب",
    },
}


def _normalise_label(label: Any) -> str:
    text = str(label).strip().lower()
    for char in (" ", "-", "/", "."):
        text = text.replace(char, "_")
    return text


def canonical_category(label: Any) -> str | None:
    normalised = _normalise_label(label)
    for category, aliases in _LABEL_ALIASES.items():
        if normalised in aliases or normalised == category:
            return category
    return None


def category_meta(key: str) -> dict[str, str]:
    for item in config.CRY_CATEGORIES:
        if item["key"] == key:
            return item
    return {"key": key, "label_en": key, "label_ar": key}


@dataclass
class Prediction:
    category: str | None
    raw_label: str
    confidence: float | None
    probabilities: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ModelRegistry:
    """Lazily loads and caches the WAFAA AI model, watching for file changes."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._meta: dict[str, Any] = {}
        self._loaded_mtime: float | None = None
        self._load_error: dict[str, str] | None = None
        self._sha256: str | None = None

    # --- loading ---------------------------------------------------------
    def _file_signature(self) -> float | None:
        try:
            return config.MODEL_PATH.stat().st_mtime
        except OSError:
            return None

    def _read_meta(self) -> dict[str, Any]:
        try:
            raw = config.MODEL_META_PATH.read_text(encoding="utf-8")
        except OSError:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _sha256_of_model(self) -> str | None:
        try:
            digest = hashlib.sha256()
            with config.MODEL_PATH.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError:
            return None

    def _check_digest_pin(self) -> dict[str, str] | None:
        expected = os.environ.get("WAFAA_MODEL_SHA256", "").strip().lower()
        if not expected:
            return None
        actual = self._sha256_of_model()
        if actual != expected:
            return {
                "code": "model_digest_mismatch",
                "message": (
                    "The model file digest does not match WAFAA_MODEL_SHA256. "
                    "Loading was refused for safety."
                ),
            }
        return None

    def _load(self) -> None:
        if not config.MODEL_PATH.exists():
            self._model = None
            self._load_error = None
            self._meta = {}
            self._sha256 = None
            self._loaded_mtime = None
            return

        digest_error = self._check_digest_pin()
        if digest_error:
            self._model = None
            self._load_error = digest_error
            return

        self._meta = self._read_meta()
        try:
            import joblib  # imported lazily so the API can start without it

            self._model = joblib.load(config.MODEL_PATH)
            self._load_error = None
            self._sha256 = self._sha256_of_model()
            self._loaded_mtime = self._file_signature()
        except ModuleNotFoundError as exc:
            self._model = None
            self._load_error = {
                "code": "model_missing_dependency",
                "message": (
                    f"The model was found but needs the Python package "
                    f"'{exc.name}' to be loaded. Install it with "
                    f"`pip install -r backend/requirements-model.txt` and restart "
                    "the API."
                ),
            }
        except Exception as exc:  # noqa: BLE001 - surfaced to the parent
            self._model = None
            self._load_error = {
                "code": "model_load_failed",
                "message": f"The model file could not be loaded: {exc}",
            }

    def refresh(self) -> None:
        signature = self._file_signature()
        if signature is None:
            if self._model is not None or self._load_error is not None:
                self._load()
            return
        if signature != self._loaded_mtime or self._model is None:
            self._load()

    # --- introspection ---------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._model is not None

    def status(self) -> dict[str, Any]:
        with _MODEL_LOCK:
            self.refresh()
            warnings: list[str] = []
            declared_features = self._meta.get("feature_vector")
            if declared_features and declared_features != features.FEATURE_VECTOR_VERSION:
                warnings.append(
                    "The model sidecar declares feature vector "
                    f"'{declared_features}' but the API extracts "
                    f"'{features.FEATURE_VECTOR_VERSION}'. Predictions will only be "
                    "meaningful if both pipelines match."
                )
            return {
                "connected": self.connected,
                "message_code": None if self.connected else (
                    self._load_error["code"] if self._load_error else "model_not_connected"
                ),
                "message": None if self.connected else (
                    self._load_error["message"] if self._load_error else
                    "The WAFAA AI model file has not been uploaded yet."
                ),
                "model_path": str(config.MODEL_PATH),
                "model_version": self._meta.get("version"),
                "trained_at": self._meta.get("trained_at"),
                "labels": self._declared_labels(),
                "feature_vector": features.FEATURE_VECTOR_VERSION,
                "declared_feature_vector": declared_features,
                "feature_dimension": features.FEATURE_DIMENSION,
                "sample_rate": config.TARGET_SAMPLE_RATE,
                "categories": config.CRY_CATEGORIES,
                "sha256": self._sha256,
                "warnings": warnings,
                "meta_notes": self._meta.get("notes"),
            }

    def _declared_labels(self) -> list[str]:
        labels = self._meta.get("labels")
        if isinstance(labels, list) and labels:
            return [str(label) for label in labels]
        classes = getattr(self._model, "classes_", None)
        if classes is not None:
            try:
                return [str(label) for label in list(classes)]
            except TypeError:
                return []
        return []

    # --- inference -------------------------------------------------------
    def predict(self, feature_vector: np.ndarray) -> Prediction:
        with _MODEL_LOCK:
            self.refresh()
            if self._model is None:
                raise ModelNotConnected(
                    self._load_error["message"] if self._load_error else
                    "The WAFAA AI model file has not been uploaded yet.",
                    self._load_error["code"] if self._load_error else "model_not_connected",
                )

            model = self._model
            declared_labels = self._declared_labels()
            warnings: list[str] = []
            declared_features = self._meta.get("feature_vector")
            if declared_features and declared_features != features.FEATURE_VECTOR_VERSION:
                warnings.append(
                    f"Feature vector mismatch: model declares '{declared_features}', "
                    f"API extracts '{features.FEATURE_VECTOR_VERSION}'."
                )

            matrix = feature_vector.reshape(1, -1)
            probabilities: list[dict[str, Any]] = []
            confidence: float | None = None
            raw_label: str | None = None

            predict_proba = getattr(model, "predict_proba", None)
            if callable(predict_proba):
                try:
                    proba = np.asarray(predict_proba(matrix), dtype=float)
                except Exception as exc:  # noqa: BLE001
                    raise InferenceError(
                        "The model rejected the extracted features: "
                        f"{exc}. Check that preprocessing matches training."
                    ) from exc
                if proba.ndim == 2 and proba.shape[0] == 1:
                    proba = proba[0]
                if proba.ndim == 1 and proba.size:
                    if np.all(np.isfinite(proba)) and np.all(proba >= 0) and np.all(proba <= 1):
                        total = float(proba.sum())
                        if 0.98 <= total <= 1.02:
                            labels = declared_labels or [
                                f"class_{index}" for index in range(proba.size)
                            ]
                            if len(labels) != proba.size:
                                warnings.append(
                                    "Model class labels do not match the number of "
                                    "probabilities; confidence was not reported."
                                )
                            else:
                                order = np.argsort(proba)[::-1]
                                probabilities = [
                                    {
                                        "raw_label": labels[int(index)],
                                        "category": canonical_category(labels[int(index)]),
                                        "probability": round(float(proba[int(index)]), 4),
                                    }
                                    for index in order
                                ]
                                best = int(order[0])
                                raw_label = labels[best]
                                confidence = round(float(proba[best]), 4)
                        else:
                            warnings.append(
                                "The model probability output is not normalised, so no "
                                "confidence score is shown."
                            )
                else:
                    warnings.append(
                        "The model returned an unexpected probability shape, so no "
                        "confidence score is shown."
                    )

            if raw_label is None:
                predict = getattr(model, "predict", None)
                if not callable(predict):
                    raise InferenceError(
                        "The loaded object does not expose predict() or "
                        "predict_proba(), so it cannot be used for analysis."
                    )
                try:
                    output = predict(matrix)
                except Exception as exc:  # noqa: BLE001
                    raise InferenceError(
                        "The model rejected the extracted features: "
                        f"{exc}. Check that preprocessing matches training."
                    ) from exc
                raw_label = str(np.asarray(output).reshape(-1)[0])
                warnings.append(
                    "This model does not provide probability estimates, so no "
                    "confidence score is shown."
                )

            category = canonical_category(raw_label)
            if category is None:
                warnings.append(
                    f"The model returned the label '{raw_label}', which is not one of "
                    "the five WAFAA AI categories. Add it to the sidecar label map to "
                    "display a translated name."
                )
            return Prediction(
                category=category,
                raw_label=raw_label,
                confidence=confidence,
                probabilities=probabilities,
                warnings=warnings,
            )


class ModelNotConnected(Exception):
    def __init__(self, message: str, code: str = "model_not_connected") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class InferenceError(Exception):
    """The model is loaded but inference failed (e.g. feature mismatch)."""


registry = ModelRegistry()
