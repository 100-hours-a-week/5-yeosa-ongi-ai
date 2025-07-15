import logging
from typing import List

from app.schemas.kafka.score import ScoreKafkaRequest, ScoreKafkaResponse
from app.schemas.models.score import ScoreResponse
from app.service.highlight_pipeline import run_highlight_pipeline
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)


async def handle(messages: List[ScoreKafkaRequest]) -> List[ScoreKafkaResponse]:
    responses: List[ScoreKafkaResponse] = []

    for msg in messages:
        task_id = msg.taskId
        album_id = msg.albumId
        raw_categories = msg.categories

        if not task_id or not album_id or not isinstance(raw_categories, list) or not raw_categories:
            logger.warning(f"[INVALID] 요청 필드 누락 또는 형식 오류: {msg}")
            status_code = 400
            responses.append(ScoreKafkaResponse(
                taskId=task_id or "unknown",
                albumId=album_id or -1,
                statusCode=status_code,
                body=ScoreResponse(
                    message=get_message_by_status(status_code),
                    data=None
                )
            ))
            continue

        try:
            status_code, response_body = await run_highlight_pipeline(msg)

            response = ScoreKafkaResponse(
                taskId=task_id,
                albumId=album_id,
                statusCode=status_code,
                body=response_body
            )
            responses.append(response)

        except Exception:
            logger.exception(f"[SCORE_HANDLE] 메시지 처리 중 예외 발생:  task_id={task_id}")

            status_code = 500
            error_response = ScoreKafkaResponse(
                taskId=task_id,
                albumId=album_id,
                statusCode=status_code,
                body=ScoreResponse(
                    message=get_message_by_status(status_code),
                    data=None
                )
            )
            responses.append(error_response)

    return responses
