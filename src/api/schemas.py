from pydantic import BaseModel
from typing import List

class HeartDiseaseRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
