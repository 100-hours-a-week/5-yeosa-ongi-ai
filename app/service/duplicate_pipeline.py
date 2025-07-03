import logging
from functools import partial
from typing import Tuple

from app.config.app_config import get_config
from app.schemas.common.request import ImageRequest
from app.schemas.models.duplicate import DuplicateResponse, DuplicateMultiResponseData
from app.service.duplicate import find_duplicate_groups
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)


async def run_duplicate_pipeline(req: ImageRequest) -> Tuple[int, DuplicateResponse]:
    """
    중복 이미지 그룹화 파이프라인 (HTTP/Kafka 공용)

    Args:
        req (ImageRequest): 이미지 파일명 목록 포함 요청

    Returns:
        Tuple[int, DuplicateResponse]: 상태 코드와 Pydantic 응답 모델
    """
    try:
        config = get_config()
        loop = config.loop
        image_refs = req.images

        if not image_refs:
            logger.warning("[DUPLICATE_PIPELINE] 입력 이미지 없음")
            status_code = 400
            return status_code, DuplicateResponse(
                message=get_message_by_status(status_code),
                data=None
            )

        # 이미지 로드
        image_loader = config.image_loader
        images = await image_loader.load_images(image_refs, "GRAY")

        logger.debug(
            "[DUPLICATE_PIPELINE] 이미지 로드 완료",
            extra={"loaded_images": len(images)},
        )

        # 중복 그룹 검색
        task_func = partial(find_duplicate_groups, images, image_refs)
        duplicate_groups = await loop.run_in_executor(None, task_func)

        # 로그 출력
        total_duplicates = sum(len(group) for group in duplicate_groups)
        logger.info(
            "[DUPLICATE_PIPELINE_DONE] 그룹 수=%d, 총 중복 이미지 수=%d",
            len(duplicate_groups),
            total_duplicates,
        )

        status_code = 201
        data = DuplicateMultiResponseData(duplicate_images=duplicate_groups)
        return status_code, DuplicateResponse(
            message=get_message_by_status(status_code),
            data=data.result()
        )

    except Exception:
        logger.exception("[INTERNAL_ERROR] 중복 이미지 파이프라인 처리 중 예외 발생")
        return 500, DuplicateResponse(
            message="internal_server_error",
            data=None
        )
