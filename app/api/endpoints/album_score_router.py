import asyncio

from fastapi import APIRouter, Request

from app.api.controllers.album_score_controller import (
    highlight_scoring_controller,
)
from app.schemas.album_schema import CategoryScoreRequest
from app.utils.logging_decorator import log_flow

router = APIRouter(tags=["score"])

# TODO: semaphoe 개수 설정
SCORE_SEMAPHORE_SIZE = 5
score_semaphore = asyncio.Semaphore(SCORE_SEMAPHORE_SIZE)

@router.post("", status_code=201)
@log_flow
async def highlight_scoring(req: CategoryScoreRequest, request: Request):
    async with score_semaphore:
        return await highlight_scoring_controller(req, request)
