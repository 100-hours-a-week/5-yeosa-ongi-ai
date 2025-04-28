from fastapi import APIRouter, Request
from app.api.controllers.album_score_controller import (
    highlight_scoring_controller,
)
from app.schemas.album_schema import CategoryScoreRequest

router = APIRouter(tags=["score"])


@router.post("/", status_code=201)
def highlight_scoring(req: CategoryScoreRequest, request: Request):
    return highlight_scoring_controller(req, request)
