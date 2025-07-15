import logging
from fastapi.responses import JSONResponse

from app.schemas.http.people import PeopleHttpRequest, PeopleHttpResponse
from app.service.people_pipeline import run_people_clustering_pipeline
from app.utils.logging_decorator import log_flow
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)

@log_flow
async def people_controller(req: PeopleHttpRequest) -> JSONResponse:
    """
    클라이언트로부터 이미지 파일명을 받아 GPU 서버에 전달하고,
    클러스터링 결과를 받아 응답합니다.

    Args:
        req (ImageRequest): 이미지 파일 이름 리스트를 포함한 요청 객체입니다.

    Returns:
        dict: 클러스터링 결과를 포함한 응답. 예시:
            {
                "message": "success",
                "data": [
                    ["img001.jpg", "img005.jpg"],  # 동일 인물 A
                    ["img002.jpg"],                # 동일 인물 B
                    ...
                ]
            }
    """
    try:
        logger.info("인물 클러스터링 요청 처리 시작", extra={"total_images": len(req.images)})

        status_code, response = await run_people_clustering_pipeline(req)

        logger.info("인물 클러스터링 완료", extra={
            "status_code": status_code,
            "cluster_count": len(response.data or [])
        })

        return JSONResponse(
            status_code=status_code,
            content=response.model_dump()
        )

    except Exception as e:
        logger.exception(f"[INTERNAL_ERROR] people_controller 예외 발생: {e}")
        status_code = 500
        return JSONResponse(
            status_code=status_code,
            content=PeopleHttpResponse(
                message=get_message_by_status(status_code),
                data=None
            ).model_dump()
        )
