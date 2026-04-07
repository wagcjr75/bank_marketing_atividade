from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from app.config import MODEL_PATH

_model_bundle: dict[str, Any] | None = None


class ModelNotLoadedError(RuntimeError):
    pass


def load_model() -> dict[str, Any]:
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


def reset_model_cache() -> None:
    global _model_bundle
    _model_bundle = None


def is_model_loaded() -> bool:
    return _model_bundle is not None or MODEL_PATH.exists()


def get_model_info() -> dict[str, Any]:
    bundle = load_model()
    return {
        "algorithm": bundle.get("algorithm", "Unknown"),
        "features": bundle.get("features", []),
        "metrics": bundle.get("metrics", {}),
        "version": bundle.get("version", "1.0.0"),
    }


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    bundle = load_model()
    pipeline = bundle["pipeline"]
    df = pd.DataFrame([payload])

    prediction_encoded = int(pipeline.predict(df)[0])
    probability_yes = float(pipeline.predict_proba(df)[0][1])

    return {
        "prediction": "yes" if prediction_encoded == 1 else "no",
        "probability": round(probability_yes, 4),
    }
