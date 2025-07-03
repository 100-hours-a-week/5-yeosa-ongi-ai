from app.schemas.kafka.base import KafkaBaseImageRequest, KafkaResponseWrapper
from app.schemas.models.people import PeopleResponse

class PeopleKafkaRequest(KafkaBaseImageRequest):
    """Kafka용 people 요청 DTO"""
    pass

class PeopleKafkaResponse(KafkaResponseWrapper[PeopleResponse]):
    """Kafka용 people 응답 DTO"""
    pass