from app.schemas.common.request import ImageRequest
from app.schemas.models.people import PeopleResponse

class PeopleHttpRequest(ImageRequest):
    """HTTP용 people 요청 DTO"""
    pass

class PeopleHttpResponse(PeopleResponse):
    """HTTP용 people 응답 DTO"""
    pass