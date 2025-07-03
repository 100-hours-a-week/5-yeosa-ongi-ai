from app.schemas.kafka.base import KafkaBaseImageRequest, KafkaResponseWrapper
from app.schemas.models.embedding import EmbeddingResponse

class EmbeddingKafkaRequest(KafkaBaseImageRequest):
    """Kafka용 embedding 요청 DTO"""
    pass

class EmbeddingKafkaResponse(KafkaResponseWrapper[EmbeddingResponse]):
    """Kafka용 embedding 응답 DTO"""
    pass