from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import MODEL_PATH, RANDOM_STATE

try:
    from ucimlrepo import fetch_ucirepo
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "A biblioteca 'ucimlrepo' não está instalada. Rode: pip install -r requirements.txt"
    ) from exc


FEATURES = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "duration", "campaign", "previous", "poutcome"
]


def main() -> None:
    dataset = fetch_ucirepo(id=222)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    X = X[FEATURES].copy()
    y = y.map({"yes": 1, "no": 0})

    categorical_features = [
        "job", "marital", "education", "default", "housing",
        "loan", "contact", "month", "poutcome"
    ]
    numeric_features = [
        "age", "balance", "day", "duration", "campaign", "previous"
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "algorithm": "RandomForestClassifier",
            "features": FEATURES,
            "metrics": metrics,
            "version": "1.0.0",
        },
        MODEL_PATH,
    )

    print("Modelo salvo em:", MODEL_PATH)
    print("Métricas:", metrics)


if __name__ == "__main__":
    main()
