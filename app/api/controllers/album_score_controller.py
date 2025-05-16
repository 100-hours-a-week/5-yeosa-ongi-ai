import logging
from functools import partial
from itertools import chain
from typing import Dict, List, Any

import torch
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.cache import get_cached_embeddings_parallel
from app.schemas.album_schema import CategoryScoreRequest
from app.service.highlight import score_each_category
from app.utils.logging_decorator import log_exception, log_flow

logger = logging.getLogger(__name__)


@log_flow
async def highlight_scoring_controller(req: CategoryScoreRequest, request: Request) -> JSONResponse:
    """
    각 카테고리별 이미지 점수를 계산하는 컨트롤러입니다.

    Args:
        req: 카테고리별 이미지 목록을 포함한 요청 객체
        request: FastAPI 요청 객체

    Returns:
        JSONResponse: 카테고리별 이미지 점수 정보를 포함한 응답
            {
                "message": "success",
                "data": List[Dict[str, Any]]  # 카테고리별 이미지 점수 리스트
            }

    """
    logger.info(
        "카테고리별 이미지 점수 계산 시작",
        extra={"total_categories": len(req.categories)},
    )

    # 1. 모든 이미지 목록 수집
    categories = req.categories
    all_images = list(
        chain.from_iterable(category.images for category in categories)
    )
    logger.debug(
        "이미지 목록 수집 완료",
        extra={"total_images": len(all_images)},
    )

    loop = request.app.state.loop
    
    # 2. 이미지 임베딩 로드
    logger.debug("이미지 임베딩 로드 시작")
    embed_load_func = partial(
        get_cached_embeddings_parallel,
        all_images,
    )
    image_features, missing_keys = await loop.run_in_executor(
        None,
        embed_load_func,
    )

    # 3. 임베딩이 없는 이미지 처리
    if missing_keys:
        logger.warning(
            "일부 이미지의 임베딩이 없음",
            extra={"missing_count": len(missing_keys)},
        )
        return JSONResponse(
            status_code=428,
            content={"message": "embedding_required", "data": missing_keys},
        )

    # 4. 임베딩 맵 생성
    embedding_map = {
        image: feature for image, feature in zip(all_images, image_features)
    }
    logger.debug("임베딩 맵 생성 완료")

    # 5. 카테고리별 점수 계산
    logger.debug("카테고리별 점수 계산 시작")
    aesthetic_regressor = request.app.state.aesthetic_regressor
    task_func = partial(
        score_each_category,
        categories,
        embedding_map,
        aesthetic_regressor,
    )

    data = await loop.run_in_executor(None, task_func)

    # 6. 결과 로깅 및 응답
    logger.info(
        "카테고리별 이미지 점수 계산 완료",
        extra={
            "total_categories": len(categories),
            "total_images": len(all_images),
        },
    )

    return JSONResponse(
        status_code=201,
        content={"message": "success", "data": data},
    )
