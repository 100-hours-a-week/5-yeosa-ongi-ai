from fastapi import APIRouter
from app.api.controllers.album_duplicate_controller import duplicate_controller
from app.schemas.album_schema import ImageRequest

router = APIRouter(tags=["duplicate"])


@router.post("/", status_code=201)
def duplicate(req: ImageRequest):
    return duplicate_controller(req)
