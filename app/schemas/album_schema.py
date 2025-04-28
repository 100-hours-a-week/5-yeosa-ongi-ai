from pydantic import BaseModel, HttpUrl, constr
from typing import List

# /embeddings, /duplicates, /quality, /categories, /people 등에서 공통 사용
class ImageRequest(BaseModel):
    images: List[str]

# /scores 요청용
class ImageCategoryGroup(BaseModel):
    category: str
    images: List[str]
    
class CategoryScoreRequest(BaseModel):
    categories: List[ImageCategoryGroup]