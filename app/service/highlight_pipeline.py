from typing import Tuple
from itertools import chain
from functools import partial
import logging

from app.config.app_config import get_config
from app.core.cache import get_cached_embeddings_parallel
from app.service.highlight import score_each_category
from app.schemas.common.request import CategoryScoreRequest
from app.schemas.models.score import ScoreResponse, ScoreMultiResponseData
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)


async def run_highlight_pipeline(req: CategoryScoreRequest) -> Tuple[int, ScoreResponse]:
    """
    카테고리별 이미지 점수를 계산하는 공통 파이프라인 함수입니다.

    Args:
        req: 카테고리별 이미지 목록을 포함한 요청 객체

    Returns:
        Tuple[int, ScoreResponse]: (HTTP 상태 코드, 응답 모델 객체)
    """
    try:
        config = get_config()
        loop = config.loop
        categories = req.categories

        # 이미지 수집
        all_images = list(
            chain.from_iterable(category.images for category in categories)
        )

        # 임베딩 로딩
        image_features, missing_keys = await get_cached_embeddings_parallel(all_images)

        if missing_keys:
            logger.warning(f"[EMBEDDING_REQUIRED] 누락된 임베딩: {missing_keys}")
            status_code = 428
            data = ScoreMultiResponseData(invalid_images=missing_keys)

            return status_code, ScoreResponse(
                message=get_message_by_status(status_code),
                data=data.result()
            )

        # 임베딩 맵
        embedding_map = {
            image: feature for image, feature in zip(all_images, image_features)
        }

        # 점수 계산
        aesthetic_regressor = config.aesthetic_regressor
        task_func = partial(
            score_each_category,
            categories,
            embedding_map,
            aesthetic_regressor,
        )
        scored_data = await loop.run_in_executor(None, task_func)
        data = ScoreMultiResponseData(score_category_clusters=scored_data)

        status_code = 201
        return status_code, ScoreResponse(
            message=get_message_by_status(status_code),
            data=data.result()
        )

    except Exception as e:
        logger.exception("[INTERNAL_ERROR] Score 파이프라인 처리 중 예외 발생")
        status_code = 500
        return status_code, ScoreResponse(
            message=get_message_by_status(status_code),
            data=None
        )