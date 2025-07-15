from app.schemas.kafka.base import KafkaBaseImageRequest, KafkaResponseWrapper
from app.schemas.models.quality import QualityResponse

class QualityKafkaRequest(KafkaBaseImageRequest):
    """Kafka용 quality 요청 DTO"""
    pass

class QualityKafkaResponse(KafkaResponseWrapper[QualityResponse]):
    """Kafka용 quality 응답 DTO"""
    pass