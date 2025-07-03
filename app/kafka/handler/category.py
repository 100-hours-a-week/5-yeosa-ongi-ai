import logging
from typing import List

from app.service.category_pipeline import run_category_pipeline
from app.schemas.models.categories import CategoriesResponse
from app.schemas.kafka.categories import CategoriesKafkaRequest, CategoriesKafkaResponse
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)


async def handle(messages: List[CategoriesKafkaRequest]) -> List[CategoriesKafkaResponse]:
    responses: List[CategoriesKafkaResponse] = []

    for msg in messages:
        task_id = msg.taskId
        album_id = msg.albumId
        image_refs = msg.images

        if not task_id or not album_id or not image_refs:
            logger.warning(f"[INVALID] 필드 누락 또는 형식 오류: task_id={task_id}, album_id={album_id}")
            responses.append(CategoriesKafkaResponse(
                taskId=task_id or "unknown",
                albumId=album_id or -1,
                statusCode=400,
                body=CategoriesResponse(
                    message="invalid_request",
                    data=None
                )
            ))
            continue

        try:
            status_code, response_body = await run_category_pipeline(msg)
            response = CategoriesKafkaResponse(
                taskId=msg.taskId,
                albumId=msg.albumId,
                statusCode=status_code,
                body=response_body
            )
            responses.append(response)

        except Exception as e:
            logger.exception(f"[CATEGORY_HANDLE] 메시지 처리 중 예외 발생: taskId={msg.taskId}")

            status_code = 500
            error_response = CategoriesKafkaResponse(
                taskId=msg.taskId,
                albumId=msg.albumId,
                statusCode=status_code,
                body=CategoriesResponse( 
                    message=get_message_by_status(status_code),
                    data=None
                )
            )
            responses.append(error_response)

    return responses