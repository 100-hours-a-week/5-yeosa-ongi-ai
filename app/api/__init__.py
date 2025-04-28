from fastapi import APIRouter
from app.api.endpoints import (
    album_category_router,
    album_duplicate_router,
    album_embedding_router,
    album_score_router,
)

api_router = APIRouter()

api_router.include_router(
    album_category_router.router,
    prefix="/api/albums/categories",
    tags=["categories"],
)
api_router.include_router(
    album_duplicate_router.router,
    prefix="/api/albums/duplicates",
    tags=["duplicates"],
)
api_router.include_router(
    album_embedding_router.router,
    prefix="/api/albums/embedding",
    tags=["embedding"],
)
api_router.include_router(
    album_score_router.router, prefix="/api/albums/score", tags=["scores"]
)
