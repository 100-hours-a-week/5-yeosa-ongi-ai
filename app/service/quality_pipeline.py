# app/pipeline/quality_pipeline.py

import asyncio
import logging
from typing import Tuple

from app.config.app_config import get_config
from app.service.quality import get_clip_low_quality_images, get_laplacian_low_quality_images
from app.schemas.common.request import ImageRequest
from app.schemas.models.quality import QualityResponse, QualityMultiResponseData
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)
THRESHOLD = 80


async def run_quality_pipeline(req: ImageRequest) -> Tuple[int, QualityResponse]:
    """
    저품질 이미지 판별 파이프라인

    Returns:
        Tuple[int, QualityResponse]: (상태 코드, 응답 DTO)
    """
    try:
        config = get_config()
        image_refs = req.images
        image_loader = config.image_loader
        text_features = config.quality_text_features
        fields = config.quality_fields

        laplacian_task = asyncio.create_task(
            get_laplacian_low_quality_images(image_refs, image_loader, THRESHOLD)
        )
        clip_task = asyncio.create_task(
            get_clip_low_quality_images(image_refs, text_features, fields)
        )

        await asyncio.wait([laplacian_task, clip_task], return_when=asyncio.FIRST_COMPLETED)

        clip_low_images, missing_keys = await clip_task

        if missing_keys:
            # embedding 누락 → 라플라시안 작업 중단
            laplacian_task.cancel()
            try:
                await laplacian_task
            except asyncio.CancelledError:
                logger.debug("laplacian_task cancelled")

            status_code = 428
            data = QualityMultiResponseData(invalid_images=missing_keys)
            return status_code, QualityResponse(
                message=get_message_by_status(status_code),
                data=data.result(),
            )

        # 둘 다 완료 시
        laplacian_low_images = await laplacian_task
        low_quality_result = list(set(laplacian_low_images) | set(clip_low_images))

        status_code = 201
        data = QualityMultiResponseData(low_quality_images=low_quality_result)

        return status_code, QualityResponse(
            message=get_message_by_status(status_code),
            data=data.result()
        )

    except Exception:
        logger.exception("[INTERNAL_ERROR] 저품질 판별 파이프라인 중 예외 발생")
        status_code = 500
        return status_code, QualityResponse(
            message=get_message_by_status(status_code),
            data=None,
        )
