from typing import Optional
from pydantic import BaseModel
from app.schemas.common.response import BaseResponse

class DuplicateMultiResponseData(BaseModel):
    duplicate_images: Optional[list[list[str]]] = None
    invalid_images: Optional[list[str]] = None

    def result(self) -> list:
        return self.duplicate_images or self. invalid_images or []
    
class DuplicateResponse(BaseResponse):
    """duplicate 응답 DTO"""
    data: Optional[list] = None