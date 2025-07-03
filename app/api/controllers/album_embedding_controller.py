import logging
from fastapi.responses import JSONResponse

from app.schemas.http.embedding import EmbeddingHttpRequest, EmbeddingHttpResponse
from app.service.embedding_pipeline import run_embedding_pipeline
from app.utils.logging_decorator import log_flow
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)

@log_flow
async def embed_controller(req: EmbeddingHttpRequest) -> JSONResponse:
    """
    클라이언트로부터 이미지 파일명을 받아 GPU 서버에 전달하고,
    임베딩 결과를 받아 캐싱하는 컨트롤러입니다.
    """
    try:
        logger.info("임베딩 요청 처리 시작", extra={"total_images": len(req.images)})

        status_code, response = await run_embedding_pipeline(req)

        logger.info("임베딩 완료", extra={
            "status_code": status_code,
            "invalid_images": len(response.data.invalid_images or []) if response.data else 0
        })

        return JSONResponse(
            status_code=status_code,
            content=response.model_dump()
        )

    except Exception as e:
        logger.exception(f"[INTERNAL_ERROR] Embedding Http 컨트롤러 예외 발생: {3}")
        status_code = 500
        return JSONResponse(
            status_code=status_code,
            content=EmbeddingHttpResponse(
                message=get_message_by_status(status_code),
                data=None
            ).model_dump()
        )
