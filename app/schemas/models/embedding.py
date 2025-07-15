from typing import Optional
from pydantic import BaseModel, field_serializer
from app.schemas.common.response import BaseResponse

class EmbeddingMultiResponseData(BaseModel):
    invalid_images: Optional[list[str]] = None

    def result(self):
        return self.invalid_images or []
    
class EmbeddingResponse(BaseResponse):
    """embedding 응답 DTO"""
    data: Optional[list] = None