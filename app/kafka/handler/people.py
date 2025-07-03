import logging
from typing import List

from app.schemas.kafka.people import PeopleKafkaRequest, PeopleKafkaResponse
from app.schemas.models.people import PeopleResponse
from app.service.people_pipeline import run_people_clustering_pipeline
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)


async def handle(messages: List[PeopleKafkaRequest]) -> List[PeopleKafkaResponse]:
    """
    Kafka에서 받은 이미지 리스트를 GPU 서버로 전달하고,
    클러스터링 결과를 응답 메시지 리스트로 반환합니다.
    """
    responses: List[PeopleKafkaResponse] = []

    for msg in messages:
        task_id = msg.taskId
        album_id = msg.albumId
        image_refs = msg.images

        if not task_id or not album_id or not image_refs:
            logger.warning(f"[INVALID] 필드 누락 또는 형식 오류: task_id={task_id}, album_id={album_id}")
            status_code=400
            responses.append(PeopleKafkaResponse(
                taskId=task_id or "unknown",
                albumId=album_id or -1,
                statusCode=status_code,
                body=PeopleResponse(
                    message=get_message_by_status(status_code),
                    data=None
                )
            ))
            continue

        try:
            status_code, response_body = await run_people_clustering_pipeline(msg)

            response = PeopleKafkaResponse(
                    taskId=task_id,
                    albumId=album_id,
                    statusCode=status_code,
                    body=response_body
                )
            responses.append(response)

        except Exception as e:
            logger.exception(f"[EXCEPTION] 핸들러 처리 중 오류 발생: taskId={msg.taskId}")

            status_code = 500
            error_response = PeopleKafkaResponse(
                taskId=task_id or "unknown",
                albumId=album_id or -1,
                statusCode=status_code,
                body=PeopleResponse(
                    message=get_message_by_status(status_code),
                    data=None
                )
            )
            responses.append(error_response)

    return responses