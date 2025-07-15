from app.schemas.common.request import ImageRequest
from app.schemas.models.quality import QualityResponse

class QualityHttpRequest(ImageRequest):
    """HTTP용 quality 요청 DTO"""
    pass

class QualityHttpResponse(QualityResponse):
    """HTTP용 quality 응답 DTO"""
    pass