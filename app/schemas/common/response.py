from pydantic import BaseModel, field_serializer
from typing import Any, Optional

class BaseResponse(BaseModel):
    message: str
    data: Optional[Any] = None
