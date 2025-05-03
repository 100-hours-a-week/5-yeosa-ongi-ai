from fastapi import APIRouter
from app.api.controllers.album_quality_controller import quality_controller
from app.schemas.album_schema import ImageRequest
from app.utils.logging_utils import log_flow
from fastapi import Request

router = APIRouter(tags=["category"])

@router.post("/", status_code=201)
@log_flow
def quality(req: ImageRequest, request: Request):
    return quality_controller(req, request)
