from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from .feature_extraction import extract_features

DEFAULT_SPEAKER_MODEL = Path("artifacts/speaker_model/speaker_model.pkl")

_speaker_lock = threading.Lock()
_speaker_cache: dict[str, Any] = {}


def load_speaker_bundle(model_path: Path | str = DEFAULT_SPEAKER_MODEL):
    model_path = Path(model_path)
    if not model_path.exists():
        return None

    cache_key = str(model_path.resolve())
    with _speaker_lock:
        if cache_key not in _speaker_cache:
            try:
                import joblib
            except ImportError as exc:
                raise RuntimeError(
                    "Speaker recognition needs joblib. Install dependencies with: "
                    "pip install -r requirements.txt"
                ) from exc

            _speaker_cache[cache_key] = joblib.load(model_path)
        return _speaker_cache[cache_key]


def predict_speaker(audio_path: Path | str, model_path: Path | str = DEFAULT_SPEAKER_MODEL) -> dict[str, str | float | None]:
    try:
        bundle = load_speaker_bundle(model_path)
    except RuntimeError:
        return {
            "speaker_id": None,
            "speaker_confidence": None,
            "speaker_status": "dependency_missing",
        }

    if bundle is None:
        return {
            "speaker_id": None,
            "speaker_confidence": None,
            "speaker_status": "model_missing",
        }

    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
    try:
        features = extract_features(audio_path)
    except RuntimeError:
        return {
            "speaker_id": None,
            "speaker_confidence": None,
            "speaker_status": "dependency_missing",
        }

    if features is None:
        return {
            "speaker_id": None,
            "speaker_confidence": None,
            "speaker_status": "feature_error",
        }

    features = features.reshape(1, -1)
    speaker_id = str(model.predict(features)[0])
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))

    return {
        "speaker_id": speaker_id,
        "speaker_confidence": confidence,
        "speaker_status": "ok",
    }
