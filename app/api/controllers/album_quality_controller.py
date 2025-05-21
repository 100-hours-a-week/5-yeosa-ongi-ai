import logging, asyncio
import asyncio

from fastapi import Request
from fastapi.responses import JSONResponse

from app.schemas.album_schema import ImageRequest
from app.service.quality import get_clip_low_quality_images, get_laplacian_low_quality_images
from app.utils.logging_decorator import log_exception, log_flow

logger = logging.getLogger(__name__)
THRESHOLD = 80  # 저품질 이미지 판별을 위한 임계값


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

    image_refs = req.images
    image_loader = request.app.state.image_loader
    text_features = request.app.state.quality_text_features
    fields = request.app.state.quality_fields
    
    laplacian_task = asyncio.create_task(get_laplacian_low_quality_images(image_refs, image_loader, THRESHOLD))
    clip_task = asyncio.create_task(get_clip_low_quality_images(image_refs, text_features, fields))
    
    await asyncio.wait([clip_task, laplacian_task], return_when=asyncio.FIRST_COMPLETED)
    
    clip_low_images, missing_keys = await clip_task

    if missing_keys:
        laplacian_task.cancel()
        try:
            await laplacian_task
        except asyncio.CancelledError:
            logger.debug("laplacian_task cancelled")
        return JSONResponse(
            status_code=428,
            content={"message": "embedding_required", "data": missing_keys},
        )
            
    laplacian_low_images = await laplacian_task
        
    result = list(set(laplacian_low_images) | set(clip_low_images))

    logger.info(
        "저품질 이미지 검색 완료",
        extra={
            "total_images": len(image_refs),
            "low_quality_count": len(result),
        },
    )

    return JSONResponse(
        status_code=201,
        content={"message": "success", "data": result},
    )


