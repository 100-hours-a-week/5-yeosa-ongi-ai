from fastapi import APIRouter
from app.api.controllers.album_category_controller import categorize_controller
from app.schemas.album_schema import ImageRequest
from fastapi import Request

router = APIRouter(tags=["category"])


@router.post("/", status_code=201)
def categorize(req: ImageRequest, request: Request):
    return categorize_controller(req, request)
