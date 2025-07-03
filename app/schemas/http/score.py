from app.schemas.common.request import CategoryScoreRequest
from app.schemas.models.score import ScoreResponse

class ScoreHttpRequest(CategoryScoreRequest):
    """HTTP용 score 요청 DTO"""
    pass

class ScoreHttpResponse(ScoreResponse):
    """HTTP용 score 응답 DTO"""
    pass