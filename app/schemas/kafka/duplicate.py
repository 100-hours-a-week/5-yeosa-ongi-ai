from app.schemas.kafka.base import KafkaBaseImageRequest, KafkaResponseWrapper
from app.schemas.models.duplicate import DuplicateResponse

class DuplicateKafkaRequest(KafkaBaseImageRequest):
    """Kafka용 duplicate 요청 DTO"""
    pass

class DuplicateKafkaResponse(KafkaResponseWrapper[DuplicateResponse]):
    """Kafka용 duplicate 응답 DTO"""
    pass