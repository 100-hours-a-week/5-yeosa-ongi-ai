from fastapi import APIRouter, Request

from app.api.controllers.album_score_controller import (
    highlight_scoring_controller,
)
from app.schemas.album_schema import CategoryScoreRequest
from app.utils.logging_decorator import log_flow

router = APIRouter(tags=["score"])


@router.post("", status_code=201)
@log_flow
async def highlight_scoring(req: CategoryScoreRequest, request: Request):
    return await request.app.state.postprocess_queue.enqueue(
        lambda: highlight_scoring_controller(req, request)
    )
