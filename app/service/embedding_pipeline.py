import pickle
import logging
from typing import Tuple

from app.schemas.common.request import ImageRequest
from app.schemas.models.embedding import EmbeddingResponse, EmbeddingMultiResponseData
from app.core.cache import set_cached_embedding
from app.config.app_config import get_config
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)


async def run_embedding_pipeline(req: ImageRequest) -> Tuple[int, EmbeddingResponse]:
    """
    이미지 임베딩 요청 및 캐싱 파이프라인 (HTTP/Kafka 공용)

    Args:
        req (ImageRequest): 이미지 파일명 목록 포함 요청

    Returns:
        Tuple[int, EmbeddingResponse]: 상태 코드와 응답 모델
    """
    try:
        config = get_config()
        gpu_client = config.gpu_client

        image_list = req.images
        if not image_list:
            status_code = 400
            logger.warning("[EMBEDDING_PIPELINE] 입력 이미지 없음")
            return status_code, EmbeddingResponse(
                message=get_message_by_status(status_code),
                data=None
            )

        # GPU 서버 POST 요청
        response = await gpu_client.post(
            "/clip/embedding",
            json={"images": image_list},
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            logger.error(f"[GPU FAIL] 상태 코드={response.status_code}")
            status_code = 500
            return status_code, EmbeddingResponse(
                message=get_message_by_status(status_code),
                data=None
            )

        result_obj = pickle.loads(await response.aread())

        result: dict[str, list[float]] = result_obj.get("data", {})

        invalid_images: list[str] = []
        for filename in image_list:
            if filename not in result:
                invalid_images.append(filename)

        for filename, feature in result.items():
            try:
                await set_cached_embedding(filename, feature)
            except Exception as e:
                logger.error(f"[Redis SET ERROR] key='{filename}' 실패: {e}", exc_info=True)
                status_code = 500
                data=EmbeddingMultiResponseData(invalid_images=invalid_images)
                return status_code, EmbeddingResponse(
                    message=get_message_by_status(status_code),
                    data=data.result()
                )
            
        status_code = 201
        data = EmbeddingMultiResponseData(invalid_images=invalid_images)
        return status_code, EmbeddingResponse(
            message=get_message_by_status(status_code),
            data=None
        )

    except Exception:
        status_code = 500
        logger.exception("[INTERNAL_ERROR] 임베딩 파이프라인 처리 중 예외 발생")
        return status_code, EmbeddingResponse(
            message=get_message_by_status(status_code),
            data=None
        )
