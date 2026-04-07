from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=18, le=100)
    job: Literal[
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown"
    ]
    marital: Literal["divorced", "married", "single"]
    education: Literal["primary", "secondary", "tertiary", "unknown"]
    default: Literal["yes", "no"]
    balance: float = Field(..., ge=-100000.0, le=1000000.0)
    housing: Literal["yes", "no"]
    loan: Literal["yes", "no"]
    contact: Literal["cellular", "telephone", "unknown"]
    day: int = Field(..., ge=1, le=31)
    month: Literal[
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]
    duration: int = Field(..., ge=0, le=5000)
    campaign: int = Field(..., ge=1, le=100)
    previous: int = Field(..., ge=0, le=100)
    poutcome: Literal["failure", "other", "success", "unknown"]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class InfoResponse(BaseModel):
    algorithm: str
    features: list[str]
    metrics: dict[str, float]
    version: str


class PredictionResponse(BaseModel):
    prediction: Literal["yes", "no"]
    probability: float = Field(..., ge=0.0, le=1.0)
