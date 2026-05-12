from pydantic import BaseModel
from typing import Optional, List

class PredictionResponse(BaseModel):
    class_id: int
    confidence: float
    class_name: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
