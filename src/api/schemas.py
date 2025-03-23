from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class AnalyticsRequest(BaseModel):
    metric: str = Field(..., example="revenue")


class QuestionRequest(BaseModel):
    question: str = Field(..., example="What is the cancellation rate?")
    max_results: int = Field(default=3, ge=1, le=10)

class AnalyticsResponse(BaseModel):
    status: str
    data: dict
    visualization: Optional[str] = None  # Base64 image

class HealthResponse(BaseModel):
    status: str
    vector_db_records: int
    llm_status: str