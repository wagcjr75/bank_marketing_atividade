from pathlib import Path

APP_NAME = "Bank Marketing Subscription Predictor"
APP_VERSION = "1.0.0"
MODEL_FILENAME = "bank_marketing_model.joblib"
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / MODEL_FILENAME
RANDOM_STATE = 42
TEST_SAMPLE = {
    "age": 35,
    "job": "admin.",
    "marital": "married",
    "education": "secondary",
    "default": "no",
    "balance": 1200.0,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 180,
    "campaign": 2,
    "previous": 0,
    "poutcome": "unknown",
}
