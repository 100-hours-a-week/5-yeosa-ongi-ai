from fastapi import APIRouter

from app.api.controllers.album_score_controller import (
    highlight_scoring_controller,
)
from app.schemas.http.score import ScoreHttpRequest
from app.utils.logging_decorator import log_flow

router = APIRouter(tags=["score"])

@router.post("", status_code=201)
@log_flow
async def highlight_scoring(req: ScoreHttpRequest):
    return await highlight_scoring_controller(req)
