from fastapi import APIRouter, Request

from app.api.controllers.album_category_controller import categorize_controller
from app.schemas.album_schema import ImageRequest
from app.utils.logging_decorator import log_flow

router = APIRouter(tags=["category"])


@router.post("", status_code=201)
@log_flow
async def categorize(req: ImageRequest, request: Request):
    return await request.app.state.postprocess_queue.enqueue(
        lambda: categorize_controller(req, request)
    )
