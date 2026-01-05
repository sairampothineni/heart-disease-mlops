from typing import List

from pydantic import BaseModel


class HeartDiseaseRequest(BaseModel):
    """
    Request schema for heart disease prediction.
    Expects a list of numerical feature values.
    """
    features: List[float]


class PredictionResponse(BaseModel):
    """
    Response schema returned by the prediction endpoint.
    """
    prediction: int
    confidence: float
