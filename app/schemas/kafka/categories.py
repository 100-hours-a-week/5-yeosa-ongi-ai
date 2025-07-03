from app.schemas.kafka.base import KafkaBaseImageConceptRequest, KafkaResponseWrapper
from app.schemas.models.categories import CategoriesResponse

class CategoriesKafkaRequest(KafkaBaseImageConceptRequest):
    """Kafka용 categories 요청 DTO"""
    pass

class CategoriesKafkaResponse(KafkaResponseWrapper[CategoriesResponse]):
    """Kafka용 categories 응답 DTO"""
    pass