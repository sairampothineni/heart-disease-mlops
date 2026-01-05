from typing import List                          # Import List type for defining list-based fields

from pydantic import BaseModel                  # BaseModel provides data validation and serialization


class HeartDiseaseRequest(BaseModel):           # Request schema for heart disease prediction input
    features: List[float]                       # List of numerical feature values sent by the client


class PredictionResponse(BaseModel):            # Response schema returned after prediction
    prediction: int                             # Predicted class (e.g., 0 = No disease, 1 = Disease)
    confidence: float                           # Confidence/probability score of the prediction
