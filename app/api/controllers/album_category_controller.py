import logging
from functools import partial
from typing import List, Dict, Any

import torch
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.cache import get_cached_embeddings_parallel
from app.schemas.album_schema import ImageCategoryGroup, ImageRequest
from app.service.category import categorize_images
from app.utils.logging_decorator import log_exception, log_flow

logger = logging.getLogger(__name__)


@log_flow
async def categorize_controller(req: ImageRequest, request: Request) -> JSONResponse:
    """
    이미지를 카테고리별로 분류하는 컨트롤러입니다.

    Args:
        req: 이미지 파일명 목록을 포함한 요청 객체
        request: FastAPI 요청 객체

    Returns:
        JSONResponse: 카테고리별 이미지 그룹 정보를 포함한 응답

    """
    logger.info(
        "이미지 카테고리 분류 요청 처리 시작",
        extra={"total_images": len(req.images)},
    )

    # 1. 상태 변수 로드
    translated_categories = request.app.state.translated_categories
    text_features = request.app.state.category_text_features
    loop = request.app.state.loop
    image_names = req.images

    # 2. 이미지 임베딩 로드
    embed_load_func = partial(
        get_cached_embeddings_parallel,
        image_names,
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

    # 4. 이미지 임베딩 정규화
    image_features = torch.stack(image_features)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # 5. 카테고리 분류
    task_func = partial(
        categorize_images,
        image_features.cpu(),
        image_names,
        text_features.cpu(),
        translated_categories,
    )
    categorized = await loop.run_in_executor(None, task_func)

    # 6. 응답 형식 변환
    response = [
        ImageCategoryGroup(category=category, images=images)
        for category, images in categorized.items()
    ]

    logger.info(
        "이미지 카테고리 분류 완료",
        extra={
            "total_images": len(image_names),
            "category_count": len(response),
            "images_per_category": {
                category: len(images)
                for category, images in categorized.items()
            },
        },
    )

    return JSONResponse(
        status_code=201,
        content={"message": "success", "data": response},
    )
