from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import APP_NAME, APP_VERSION
from app.model import get_model_info, is_model_loaded, load_model, predict
from app.schemas import HealthResponse, InfoResponse, PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except FileNotFoundError:
        pass
    yield


app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
        version=APP_VERSION,
    )


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    try:
        return InfoResponse(**get_model_info())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model not available") from exc


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(payload: PredictionRequest) -> PredictionResponse:
    try:
        result = predict(payload.model_dump())
        return PredictionResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model not available") from exc
