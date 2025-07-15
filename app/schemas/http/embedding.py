from app.schemas.common.request import ImageRequest
from app.schemas.models.embedding import EmbeddingResponse

class EmbeddingHttpRequest(ImageRequest):
    """HTTP용 embedding 요청 DTO"""
    pass

class EmbeddingHttpResponse(EmbeddingResponse):
    """HTTP용 embedding 응답 DTO"""
    pass