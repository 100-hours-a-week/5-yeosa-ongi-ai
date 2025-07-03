from typing import Generic, TypeVar
from pydantic import BaseModel
from app.schemas.common.request import ImageRequest, ImageConceptRequest, CategoryScoreRequest

T = TypeVar("T")

class KafkaMetaData(BaseModel):
    taskId: str
    albumId: int

class KafkaRequest(KafkaMetaData):
    """Kafka용 요청 메타 데이터(taskId, albumId)"""
    pass

class KafkaResponseWrapper(KafkaMetaData, Generic[T]):
    """Kafka용 응답 메타 데이터(taskId, albumId, statusCode, body)"""
    statusCode: int
    body: T

class KafkaBaseImageRequest(KafkaMetaData, ImageRequest):
    """Kafka용 Image Base 요청 DTO"""
    pass

class KafkaBaseImageConceptRequest(KafkaMetaData, ImageConceptRequest):
    """Kafka용 Image Base Concept 요청 DTO"""
    pass

class KafkaBaseCategoryScoreRequest(KafkaMetaData, CategoryScoreRequest):
    """Kafka용 Category Score 요청 DTO"""
    pass
