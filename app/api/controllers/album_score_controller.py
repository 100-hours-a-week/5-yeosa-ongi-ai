import logging

from fastapi.responses import JSONResponse

from app.schemas.http.score import ScoreHttpRequest, ScoreHttpResponse
from app.service.highlight_pipeline import run_highlight_pipeline
from app.utils.logging_decorator import log_exception, log_flow
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)


@log_flow
async def highlight_scoring_controller(req: ScoreHttpRequest) -> JSONResponse:
    """
    각 카테고리별 이미지 점수를 계산하는 컨트롤러입니다.

    Args:
        req: 카테고리별 이미지 목록을 포함한 요청 객체

    Returns:
        JSONResponse: 카테고리별 이미지 점수 정보를 포함한 응답
            {
                "message": "success",
                "data": List[Dict[str, Any]]  # 카테고리별 이미지 점수 리스트
            }

    """
    try:
        logger.info(
            "카테고리별 이미지 점수 계산 시작",
            extra={"total_categories": len(req.categories)},
        )

        status_code, response_model = await run_highlight_pipeline(req)

        logger.info(
            "카테고리별 이미지 점수 계산 완료",
            extra={
                "status_code": status_code,
                "response_size": len(response_model.data.invalid_images)
                if hasattr(response_model.data, "invalid_images")
                else len(response_model.data or []),
            },
        )

        return JSONResponse(
            status_code=status_code,
            content=response_model.dict(),
        )
    
    except Exception as e:
        logger.exception(f"[INTERNAL_ERROR] Score Http 컨트롤러 예외 발생: {3}")
        status_code = 500
        return JSONResponse(
            status_code=status_code,
            content=ScoreHttpResponse(
                message=get_message_by_status(status_code),
                data=None
            ).model_dump()
        )        
