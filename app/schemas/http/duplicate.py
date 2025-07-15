from app.schemas.common.request import ImageRequest
from app.schemas.models.duplicate import DuplicateResponse

class DuplicateHttpRequest(ImageRequest):
    """HTTP용 duplicate 요청 DTO"""
    pass

class DuplicateHttpResponse(DuplicateResponse):
    """HTTP용 duplicate 응답 DTO"""
    pass
