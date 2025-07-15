import logging
from typing import List

from app.schemas.kafka.quality import QualityKafkaRequest, QualityKafkaResponse
from app.schemas.models.quality import QualityResponse
from app.service.quality_pipeline import run_quality_pipeline
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)


async def handle(messages: List[QualityKafkaRequest]) -> List[QualityKafkaResponse]:
    responses: List[QualityKafkaResponse] = []

    for msg in messages:
        task_id = msg.taskId
        album_id = msg.albumId
        image_refs = msg.images

        if not task_id or not album_id or not isinstance(image_refs, list) or not image_refs:
            logger.warning(f"[INVALID] 필드 누락 또는 형식 오류: task_id={task_id}, album_id={album_id}")

            status_code = 400
            responses.append(QualityKafkaResponse(
                taskId=task_id or "unknown",
                albumId=album_id or -1,
                statusCode=status_code,
                body=QualityResponse(
                    message=get_message_by_status(status_code),
                    data=None
                )
            ))
            continue

        try:
            logger.info(f"[QUALITY] task_id={task_id}, album_id={album_id}, image_count={len(image_refs)}")

            status_code, response_body = await run_quality_pipeline(msg)  # msg는 ImageRequest 상속

            responses.append(QualityKafkaResponse(
                taskId=task_id,
                albumId=album_id,
                statusCode=status_code,
                body=response_body
            ))

        except Exception:
            logger.exception(f"[QUALITY_HANDLE] 메시지 처리 중 예외 발생: taskId={task_id}")

            status_code = 500
            responses.append(QualityKafkaResponse(
                taskId=task_id or "unknown",
                albumId=album_id or -1,
                statusCode=status_code,
                body=QualityResponse(
                    message=get_message_by_status(status_code),
                    data=None
                )
            ))

    return responses
