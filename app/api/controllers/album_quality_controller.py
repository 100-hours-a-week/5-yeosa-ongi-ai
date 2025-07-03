import logging

from fastapi.responses import JSONResponse

from app.schemas.http.quality import QualityHttpRequest, QualityHttpResponse
from app.service.quality_pipeline import run_quality_pipeline
from app.utils.logging_decorator import log_exception, log_flow
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)
THRESHOLD = 80  # 저품질 이미지 판별을 위한 임계값


@log_flow
async def quality_controller(req: QualityHttpRequest) -> JSONResponse:
    """
    저품질 이미지를 검색하는 컨트롤러입니다.

    Args:
        req: 이미지 파일명 목록을 포함한 요청 객체
        request: FastAPI 요청 객체

    Returns:
        JSONResponse: 저품질 이미지 목록을 포함한 응답

    """
    try:
        logger.info(
            "저품질 이미지 검색 요청 처리 시작",
            extra={"total_images": len(req.images)},
        )

        status_code, response_model = await run_quality_pipeline(req)

        logger.info(
            "저품질 이미지 검색 완료",
            extra={
                "total_images": len(req.images),
                "status_code": status_code,
            },
        )

        return JSONResponse(
            status_code=status_code,
            content=response_model.model_dump(),
        )
    
    except Exception as e:
        logger.exception(f"[INTERNAL_ERROR] quality Http 컨트롤러 예외 발생: {e}")
        status_code = 500
        return JSONResponse(
            status_code=status_code,
            content=QualityHttpResponse(
                message=get_message_by_status(status_code),
                data=None
            ).model_dump()
        )