import logging

from fastapi.responses import JSONResponse

from app.schemas.http.categories import CategoriesHttpRequest, CategoriesHttpResponse
from app.service.category_pipeline import run_category_pipeline
from app.utils.logging_decorator import log_flow
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)


@log_flow
async def categorize_controller(
    req: CategoriesHttpRequest
) -> JSONResponse:
    """
    이미지를 카테고리별로 분류하는 컨트롤러입니다.

    Args:
        req: 이미지 파일명 목록을 포함한 요청 객체
        request: FastAPI 요청 객체

    Returns:
        JSONResponse: 카테고리별 이미지 그룹 정보를 포함한 응답

    """
    try:
        logger.info(
            "카테고리 분류 요청 시작",
            extra={"total_images": len(req.images)},
        )

        status_code, response = await run_category_pipeline(req)

        logger.info("카테고리 분류 완료", extra={
            "status_code": status_code,
            "category_count": len(response.data or []) if response.data else 0
        })

        return JSONResponse(
            status_code=status_code,
            content=response.model_dump()
        )
    
    except Exception as e:
        logger.exception(f"[INTERNAL_ERROR] quality Http 컨트롤러 예외 발생: {e}")
        status_code = 500
        return JSONResponse(
            status_code=status_code,
            content=CategoriesHttpResponse(
                message=get_message_by_status(status_code),
                data=None
            ).model_dump()
        )