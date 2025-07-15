from app.schemas.kafka.base import KafkaBaseCategoryScoreRequest, KafkaResponseWrapper
from app.schemas.models.score import ScoreResponse

class ScoreKafkaRequest(KafkaBaseCategoryScoreRequest):
    """Kafka용 score 요청 DTO"""
    pass

class ScoreKafkaResponse(KafkaResponseWrapper[ScoreResponse]):
    """Kafka용 score 응답 DTO"""
    pass