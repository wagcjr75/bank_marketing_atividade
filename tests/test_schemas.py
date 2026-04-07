import pytest
from pydantic import ValidationError

from app.config import TEST_SAMPLE
from app.schemas import PredictionRequest


def test_prediction_request_valid():
    payload = PredictionRequest(**TEST_SAMPLE)
    assert payload.age == 35
    assert payload.job == "admin."


def test_prediction_request_invalid_age():
    bad = TEST_SAMPLE.copy()
    bad["age"] = 10
    with pytest.raises(ValidationError):
        PredictionRequest(**bad)


def test_prediction_request_rejects_extra_field():
    bad = TEST_SAMPLE.copy()
    bad["extra_field"] = "not allowed"
    with pytest.raises(ValidationError):
        PredictionRequest(**bad)
