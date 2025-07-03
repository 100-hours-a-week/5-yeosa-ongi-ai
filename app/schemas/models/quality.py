from typing import Optional
from pydantic import BaseModel, field_serializer
from app.schemas.common.response import BaseResponse

class QualityMultiResponseData(BaseModel):
    low_quality_images: Optional[list[str]] = None
    invalid_images: Optional[list[str]] = None

    def result(self):
        return self.low_quality_images or self. invalid_images or []
    
class QualityResponse(BaseResponse):
    """quality 응답 DTO"""
    data: Optional[list] = None