import asyncio

from fastapi import APIRouter, Request

from app.api.controllers.album_duplicate_controller import duplicate_controller
from app.schemas.album_schema import ImageRequest
from app.utils.logging_decorator import log_flow

router = APIRouter(tags=["duplicate"])

DUPLICATE_SEMAPHORE_SIZE = 5
duplicate_semaphore = asyncio.Semaphore(DUPLICATE_SEMAPHORE_SIZE)

@router.post("", status_code=201)
@log_flow
async def duplicate(req: ImageRequest, request: Request):
    async with duplicate_semaphore:
        return await duplicate_controller(req, request)

