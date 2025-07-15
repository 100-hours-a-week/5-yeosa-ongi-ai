from fastapi import APIRouter

from app.api.controllers.album_duplicate_controller import duplicate_controller
from app.schemas.http.duplicate import DuplicateHttpRequest
from app.utils.logging_decorator import log_flow

router = APIRouter(tags=["duplicate"])

@router.post("", status_code=201)
@log_flow
async def duplicate(req: DuplicateHttpRequest):
    return await duplicate_controller(req)

