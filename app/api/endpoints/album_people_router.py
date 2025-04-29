from fastapi import APIRouter
from app.api.controllers.album_people_controller import people_controller
from app.schemas.album_schema import ImageRequest
from fastapi import Request

router = APIRouter(tags=["people"])


@router.post("/", status_code=201)
def people(req: ImageRequest, request: Request):
    return people_controller(req, request)
