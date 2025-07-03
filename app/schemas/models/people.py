from typing import Optional
from pydantic import BaseModel, field_serializer
from app.schemas.common.response import BaseResponse

class RepresentativeFace(BaseModel):
    image: str
    bbox: list[float]

class PeopleCluster(BaseModel):
    images: list[str]
    representative_face: RepresentativeFace

class PeopleMultiResponseData(BaseModel):
    people_clusters: Optional[list[PeopleCluster]] = None
    invalid_images: Optional[list[str]] = None

    def result(self):
        return self.people_clusters or self.invalid_images or []
    
class PeopleResponse(BaseResponse):
    """people 응답 DTO"""
    data: Optional[list] = None