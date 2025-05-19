import logging
from functools import partial
from typing import List, Tuple, Any

import torch
from fastapi import Request
from fastapi.responses import JSONResponse

# from app.core.cache import get_cached_embeddings_parallel
from app.schemas.album_schema import ImageRequest
from app.service.duplicate import find_duplicate_groups
from app.utils.logging_decorator import log_exception, log_flow

logger = logging.getLogger(__name__)


@log_flow
async def duplicate_controller(req: ImageRequest, request: Request) -> JSONResponse:
    """
    중복 이미지를 검색하는 컨트롤러입니다.

    Args:
        req: 이미지 파일명 목록을 포함한 요청 객체
        request: FastAPI 요청 객체

    Returns:
        JSONResponse: 중복 이미지 그룹 정보를 포함한 응답
            {
                "message": "success",
                "data": List[List[str]]  # 중복 이미지 그룹 리스트
            }

    """
    logger.info(
        "중복 이미지 검색 요청 처리 시작",
        extra={"total_images": len(req.images)},
    )

    loop = request.app.state.loop
    image_refs = req.images

    # # 1. 이미지 임베딩 로드
    # logger.debug("이미지 임베딩 로드 시작")
    # image_load_func = partial(
    #     get_cached_embeddings_parallel,
    #     image_refs,
    # )
    # image_features, missing_keys = await loop.run_in_executor(
    #     None,
    #     image_load_func,
    # )

    # 2. 임베딩이 없는 이미지 처리
    # if missing_keys:
    #     logger.warning(
    #         "일부 이미지의 임베딩이 없음",
    #         extra={"missing_count": len(missing_keys)},
    #     )
    #     return JSONResponse(
    #         status_code=428,
    #         content={"message": "embedding_required", "data": missing_keys},
    #     )
    
    # 1. 이미지 로드
    image_loader = request.app.state.image_loader
    images = await image_loader.load_images(image_refs, 'GRAY')

    logger.debug(
        "이미지 로드 완료",
        extra={"loaded_images": len(images)},
    )

    # 3. 중복 이미지 검색
    logger.debug("중복 이미지 검색 시작")
    image_features = torch.stack(image_features)
    
    task_func = partial(find_duplicate_groups, image_features, image_refs)

    data = await loop.run_in_executor(None, task_func)

    # 4. 결과 로깅 및 응답
    total_duplicates = sum(len(group) for group in data)
    logger.info(
        "중복 이미지 검색 완료",
        extra={
            "total_images": len(image_refs),
            "duplicate_groups": len(data),
            "total_duplicates": total_duplicates,
            "duplicate_ratio": f"{(total_duplicates / len(image_refs)) * 100:.1f}%",
        },
    )

    return JSONResponse(
        status_code=201,
        content={"message": "success", "data": data},
    )
