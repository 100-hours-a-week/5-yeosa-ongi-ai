import logging
from typing import List

from app.service.duplicate_pipeline import run_duplicate_pipeline
from app.schemas.models.duplicate import DuplicateResponse
from app.schemas.kafka.duplicate import DuplicateKafkaRequest, DuplicateKafkaResponse
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)


async def handle(messages: List[DuplicateKafkaRequest]) -> List[DuplicateKafkaResponse]:
    responses: List[DuplicateKafkaResponse] = []

    for msg in messages:
        task_id = msg.taskId
        album_id = msg.albumId
        image_refs = msg.images

        if not task_id or not album_id or not image_refs:
            logger.warning(f"[INVALID] 필드 누락 또는 형식 오류: task_id={task_id}, album_id={album_id}")
            status_code = 400
            responses.append(DuplicateKafkaResponse(
                taskId=task_id or "unknown",
                albumId=album_id or -1,
                statusCode=status_code,
                body=DuplicateResponse(
                    message=get_message_by_status(status_code),
                    data=None
                )
            ))
            continue

        try:
            status_code, response_body = await run_duplicate_pipeline(msg)  # msg는 ImageRequest 상속

            response = DuplicateKafkaResponse(
                taskId=task_id,
                albumId=album_id,
                statusCode=status_code,
                body=response_body
            )
            responses.append(response)

        except Exception:
            logger.exception(f"[DUPLICATE_HANDLE] 메시지 처리 중 예외 발생: taskId={msg.taskId}")

            status_code = 500
            error_response = DuplicateKafkaResponse(
                taskId=task_id or "unknown",
                albumId=album_id or -1,
                statusCode=status_code,
                body=DuplicateResponse(
                    message=get_message_by_status(status_code),
                    data=None
                )
            )
            responses.append(error_response)

    return responses
