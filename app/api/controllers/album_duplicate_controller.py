import logging

from fastapi.responses import JSONResponse

# from app.core.cache import get_cached_embeddings_parallel
from app.schemas.http.duplicate import DuplicateHttpRequest, DuplicateHttpResponse
from app.service.duplicate_pipeline import run_duplicate_pipeline
from app.utils.logging_decorator import log_exception, log_flow
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)

@log_flow
async def duplicate_controller(req: DuplicateHttpRequest) -> JSONResponse:
    """
    중복 이미지를 검색하는 컨트롤러입니다.

    Args:
        req: 이미지 파일명 목록을 포함한 요청 객체

    Returns:
        JSONResponse: 중복 이미지 그룹 정보를 포함한 응답
            {
                "message": "success",
                "data": List[List[str]]  # 중복 이미지 그룹 리스트
            }

    """
    try:
        logger.info(
            "중복 이미지 검색 요청 처리 시작",
            extra={"total_images": len(req.images)},
        )

        status_code, response = await run_duplicate_pipeline(req)

        # HACK: invalid_image가 있는 경우도 카테고리 분류 완료로 찍힘
        logger.info("카테고리 분류 완료", extra={
            "status_code": status_code,
            "duplicate_count": len(response.data.duplicate_images or []) if response.data else 0
        })

        return JSONResponse(
            status_code=status_code,
            content=response.model_dump()
        )
    
    except Exception as e:
        logger.exception(f"[INTERNAL_ERROR] Duplicate Http 컨트롤러 예외 발생: {e}")
        status_code = 500
        return JSONResponse(
            status_code=status_code,
            content=DuplicateHttpResponse(
                message=get_message_by_status(status_code),
                data=None
            ).model_dump()
        )