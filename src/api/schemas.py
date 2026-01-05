from typing import List

from pydantic import BaseModel


class HeartDiseaseRequest(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
