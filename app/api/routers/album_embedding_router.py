import asyncio

from fastapi import APIRouter, Request

from app.api.controllers.album_embedding_controller import embed_controller
from app.schemas.http.embedding import EmbeddingHttpRequest
from app.utils.logging_decorator import log_flow

router = APIRouter(tags=["embedding"])

EMBEDDING_SEMAPHORE_SIZE = 4
embedding_semaphore = asyncio.Semaphore(EMBEDDING_SEMAPHORE_SIZE)

@router.post("", status_code=201)
@log_flow
async def embed(req: EmbeddingHttpRequest):
    async with embedding_semaphore:
        return await embed_controller(req)
