from typing import Optional
from pydantic import BaseModel, field_serializer
from app.schemas.common.response import BaseResponse

class CategoryCluster(BaseModel):
    category: str
    images: list[str]

class CategoriesMultiResponseData(BaseModel):
    category_clusters: Optional[list[CategoryCluster]] = None
    invalid_images: Optional[list[str]] = None

    def result(self) -> list:
        return self.category_clusters or self.invalid_images or []
    
class CategoriesResponse(BaseResponse):
    """categories 응답 DTO"""
    data: Optional[list] = None