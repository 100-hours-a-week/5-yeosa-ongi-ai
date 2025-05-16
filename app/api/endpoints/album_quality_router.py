import asyncio

from fastapi import APIRouter, Request

from app.api.controllers.album_quality_controller import quality_controller
from app.schemas.album_schema import ImageRequest
from app.utils.logging_decorator import log_flow

router = APIRouter(tags=["category"])

# TODO: semaphoe 개수 설정
QUALITY_SEMAPHORE_SIZE = 5
quality_semaphore = asyncio.Semaphore(QUALITY_SEMAPHORE_SIZE)

@router.post("", status_code=201)
@log_flow
async def quality(req: ImageRequest, request: Request):
    async with quality_semaphore:
        return await quality_controller(req, request)
