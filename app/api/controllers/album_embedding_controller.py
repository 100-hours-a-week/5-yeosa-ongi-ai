import logging
import time
import pickle

from fastapi import Request
from fastapi.responses import JSONResponse

from app.schemas.album_schema import ImageRequest
from app.utils.logging_decorator import log_flow
from app.core.cache import set_cached_embedding

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 16

@log_flow
async def embed_controller(req: ImageRequest, request: Request) -> JSONResponse:
    """
    클라이언트로부터 이미지 파일명을 받아 GPU 서버에 전달하고,
    임베딩 결과를 받아 캐싱하는 컨트롤러입니다.
    """
    logger.info("이미지 임베딩 요청 처리 시작", extra={"total_images": len(req.images)})

    try:
        gpu_client = request.app.state.gpu_client

        # ✅ GPU 서버에 JSON 요청 (파일명 목록만)
        response = await gpu_client.post(
            "/clip/embedding",
            json=req.dict(),  # {"images": [...]}
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            logger.error("GPU 서버 응답 실패", extra={"status": response.status_code})
            return JSONResponse(
                status_code=500,
                content={"message": "embedding failed (gpu error)", "data": None}
            )

        # ✅ Pickle 응답 처리
        response_obj = pickle.loads(await response.aread())  # await response.body() 도 가능
        if response_obj.get("message") != "success":
            logger.error("GPU 서버 응답 비정상", extra={"message": response_obj.get("message")})
            return JSONResponse(
                status_code=500,
                content={"message": "embedding failed (gpu processing error)", "data": None}
            )

        result = response_obj["data"]  # Dict[str, List[float]]

        logger.info("임베딩 완료", extra={"processed_images": len(result)})

        for filename, feature in result.items():
            set_cached_embedding(filename, feature)

        return JSONResponse(status_code=201, content={"message": "success", "data": None})

    except Exception as e:
        logger.error("GPU 서버 호출 예외", exc_info=True, extra={"error": str(e)})
        return JSONResponse(
            status_code=500,
            content={"message": "embedding failed (exception)", "data": None}
        )
