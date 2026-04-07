from pathlib import Path

from fastapi.testclient import TestClient

from app.config import MODEL_PATH, TEST_SAMPLE
from app.main import app
from app.model import reset_model_cache

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_valid_response():
    response = client.post("/predict", json=TEST_SAMPLE)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["yes", "no"]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_invalid_payload_returns_422():
    bad = TEST_SAMPLE.copy()
    bad["month"] = "month-invalid"
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_returns_503_when_model_missing():
    backup_path = MODEL_PATH.with_suffix(".backup")
    if backup_path.exists():
        backup_path.unlink()

    MODEL_PATH.rename(backup_path)
    reset_model_cache()
    try:
        response = client.post("/predict", json=TEST_SAMPLE)
        assert response.status_code == 503
    finally:
        backup_path.rename(MODEL_PATH)
        reset_model_cache()
