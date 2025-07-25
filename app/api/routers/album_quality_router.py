import asyncio

from fastapi import APIRouter, Request

from app.api.controllers.album_quality_controller import quality_controller
from app.schemas.http.quality import QualityHttpRequest
from app.utils.logging_decorator import log_flow

router = APIRouter(tags=["category"])

@router.post("", status_code=201)
@log_flow
async def quality(req: QualityHttpRequest):
    return await quality_controller(req)
