from app.schemas.common.request import ImageConceptRequest
from app.schemas.models.categories import CategoriesResponse

class CategoriesHttpRequest(ImageConceptRequest):
    """HTTP용 categories 요청 DTO"""
    pass

class CategoriesHttpResponse(CategoriesResponse):
    """HTTP용 categories 응답 DTO"""
    pass

