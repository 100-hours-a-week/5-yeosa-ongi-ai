from fastapi import APIRouter, Request
from app.api.controllers.album_embedding_controller import embed_controller
from app.schemas.album_schema import ImageRequest

router = APIRouter(tags=["embedding"])


@router.post("/", status_code=201)
def embed(req: ImageRequest, request: Request):
    return embed_controller(req, request)
