from fastapi import APIRouter

from app.api.controllers.album_category_controller import categorize_controller
from app.schemas.http.categories import CategoriesHttpRequest

from app.utils.logging_decorator import log_flow

router = APIRouter(tags=["category"])

@router.post("", status_code=201)
@log_flow
async def categorize(req: CategoriesHttpRequest):
    return await categorize_controller(req)