from typing import Optional
from pydantic import BaseModel, field_serializer
from app.schemas.common.response import BaseResponse

class ScoreImage(BaseModel):
    image: str
    score: float

class ScoreCategory(BaseModel):
    category: str
    images: list[ScoreImage]

class ScoreMultiResponseData(BaseModel):
    score_category_clusters: Optional[list[ScoreCategory]] = None
    invalid_images: Optional[list[str]] = None

    def result(self):
        return self.score_category_clusters or self.invalid_images or []
    
class ScoreResponse(BaseResponse):
    """score 응답 DTO"""
    data: Optional[list] = None