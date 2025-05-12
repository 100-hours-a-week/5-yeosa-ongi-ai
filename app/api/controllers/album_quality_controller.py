import logging
from functools import partial
from typing import List, Tuple, Any

import torch
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.cache import get_cached_embeddings_parallel
from app.schemas.album_schema import ImageRequest
from app.service.quality import get_low_quality_images
from app.utils.logging_decorator import log_exception, log_flow

logger = logging.getLogger(__name__)


@log_flow
async def quality_controller(req: ImageRequest, request: Request) -> JSONResponse:
    """
    저품질 이미지를 검색하는 컨트롤러입니다.

    Args:
        req: 이미지 파일명 목록을 포함한 요청 객체
        request: FastAPI 요청 객체

    Returns:
        JSONResponse: 저품질 이미지 목록을 포함한 응답

    """
    logger.info(
        "저품질 이미지 검색 요청 처리 시작",
        extra={"total_images": len(req.images)},
    )

    loop = request.app.state.loop
    image_names = req.images

    # 1. 이미지 임베딩 로드
    embed_load_func = partial(
        get_cached_embeddings_parallel,
        image_names,
    )
    image_features, missing_keys = await loop.run_in_executor(
        None,
        embed_load_func,
    )

    # 2. 임베딩이 없는 이미지 처리
    if missing_keys:
        logger.warning(
            "일부 이미지의 임베딩이 없음",
            extra={"missing_count": len(missing_keys)},
        )
        return JSONResponse(
            status_code=428,
            content={"message": "embedding_required", "data": missing_keys},
        )

    # 3. 이미지 임베딩 정규화
    image_features = torch.stack(image_features)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # 4. 저품질 이미지 검색
    text_features = request.app.state.quality_text_features
    fields = request.app.state.quality_fields

    task_func = partial(
        get_low_quality_images,
        image_names,
        image_features,
        text_features,
        fields,
    )
    result = await loop.run_in_executor(None, task_func)

    logger.info(
        "저품질 이미지 검색 완료",
        extra={
            "total_images": len(image_names),
            "low_quality_count": len(result),
        },
    )

    return JSONResponse(
        status_code=201,
        content={"message": "success", "data": result},
    )
