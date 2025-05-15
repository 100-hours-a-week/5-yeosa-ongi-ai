import asyncio

from fastapi import APIRouter, Request

from app.api.controllers.album_category_controller import categorize_controller
from app.schemas.album_schema import ImageRequest
from app.utils.logging_decorator import log_flow

router = APIRouter(tags=["category"])

CATEGORY_SEMAPHORE_SIZE = 5
category_semaphore = asyncio.Semaphore(CATEGORY_SEMAPHORE_SIZE)

@router.post("", status_code=201)
@log_flow
async def categorize(req: ImageRequest, request: Request):
    async with category_semaphore:
        return await categorize_controller(req, request)

