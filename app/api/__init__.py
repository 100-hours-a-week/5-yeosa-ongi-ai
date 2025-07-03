from fastapi import APIRouter
from app.api.routers import (
    album_category_router,
    album_duplicate_router,
    album_embedding_router,
    album_score_router,
    album_people_router,
    album_quality_router,
    # HACK: Health check용 임시 라우터
    album_health_router
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
api_router.include_router(
    album_people_router.router, prefix="/api/albums/people", tags=["people"]
)
api_router.include_router(
    album_quality_router.router, prefix="/api/albums/quality", tags=["quality"]
)

# HACK: Health check용 임시 라우터
api_router.include_router(
    album_health_router.router,
    prefix="/health/info",
    tags=["health"]
)
