import logging

from app.config.app_config import get_config
from app.schemas.common.request import ImageRequest
from app.schemas.models.people import PeopleResponse, PeopleMultiResponseData
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)



async def run_people_clustering_pipeline(req: ImageRequest) -> tuple[int, PeopleResponse]:
    """
    GPU 서버로 인물 클러스터링 요청을 보내고 결과를 반환합니다.

    Args:
        req (ImageRequest 또는 상속 객체): 이미지 목록 포함 요청 객체

    Returns:
        Tuple[int, dict]: 상태 코드와 응답 본문
    """
    try:
        config = get_config()
        gpu_client = config.gpu_client
        image_refs = req.images

        if not image_refs:
            logger.warning("[PEOPLE_PIPELINE] 입력 이미지 없음")
            status_code = 400
            return status_code, PeopleResponse(
                message=get_message_by_status(status_code),
                data=None
            )

        logger.info("[PIPELINE] 인물 클러스터링 요청 시작", extra={"total_images": len(req.images)})


        response = await gpu_client.post(
            "/people/cluster",
            json=req.dict(),
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            logger.error(f"[GPU_ERROR] 상태 코드 {response.status_code}")
            status_code = 500
            return status_code, PeopleResponse(
                message=get_message_by_status(status_code),
                data=None
            )

        response_obj = response.json()

        cluster_result = response_obj["data"]
        logger.info(f"[PIPELINE] 클러스터링 완료 - 군집 수: {len(cluster_result)}")

        status_code = 201
        data = PeopleMultiResponseData(people_clusters=cluster_result)
        return status_code, PeopleResponse(
            message=get_message_by_status(status_code),
            data=data.result()
        )
    
    except Exception as e:
        status_code = 500
        return status_code, PeopleResponse(
            message=get_message_by_status(status_code),
            data=None
        )