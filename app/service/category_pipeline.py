from functools import partial
import torch
import logging

from app.schemas.common.request import ImageConceptRequest
from app.schemas.models.categories import CategoriesResponse, CategoriesMultiResponseData, CategoryCluster
from app.config.app_config import get_config
from app.core.cache import get_cached_embeddings_parallel
from app.service.category import categorize_images
from app.utils.status_message import get_message_by_status

logger = logging.getLogger(__name__)


async def run_category_pipeline(req: ImageConceptRequest) -> tuple[int, CategoriesResponse]:
    """
    이미지 개념 기반 카테고리 분류 파이프라인 함수 (HTTP/Kafka 공용)

    Args:
        req (ImageConceptRequest): 이미지 목록 및 개념 포함 요청 모델

    Returns:
        Tuple[int, CategoriesResponse]: 상태코드와 응답 모델
    """
    try:
        config = get_config()
        loop = config.loop

        image_names = req.images
        concepts = req.concepts or []

        # 임베딩 로딩
        image_features, missing_keys = await get_cached_embeddings_parallel(image_names)

        if missing_keys:
            logger.warning(f"[EMBEDDING_REQUIRED] 누락된 임베딩: {missing_keys}")
            status_code = 428
            data = CategoriesMultiResponseData(invalid_images=missing_keys)
            return status_code, CategoriesResponse(
                message=get_message_by_status(status_code),
                data=data.result()
            )

        # 정규화
        processed = [
            torch.tensor(f, dtype=torch.float32) if isinstance(f, list) else f
            for f in image_features
        ]
        image_tensor = torch.stack(processed)
        image_tensor /= image_tensor.norm(dim=-1, keepdim=True)

        # 카테고리/임베딩 구성
        parent_categories = config.parent_categories
        parent_embeds = config.parent_embeds
        embed_dict = config.embed_dict
        category_dict = config.category_dict

        refined_categories = list(parent_categories)
        refined_embeds = list(parent_embeds)

        for concept in concepts:
            refined_categories.extend(category_dict.get(concept, []))
            refined_embeds.extend(embed_dict.get(concept, []))

        refined_embeds_tensor = torch.stack(refined_embeds, dim=0)

        # 분류 실행
        task_func = partial(
            categorize_images,
            image_tensor.cpu(),
            image_names,
            refined_embeds_tensor.cpu(),
            refined_categories,
        )
        categorized = await loop.run_in_executor(None, task_func)

        # 응답 구성
        category_clusters = [
            CategoryCluster(category=cat, images=imgs)
            for cat, imgs in categorized.items()
            if imgs
        ]

        status_code = 201
        data = CategoriesMultiResponseData(category_clusters=category_clusters)
        return status_code, CategoriesResponse(
            message=get_message_by_status(status_code),
            data=data.result()
        )
    
    except Exception as e:
        logger.exception("[INTERNAL_ERROR] Categories 파이프라인 처리 중 예외 발생")
        status_code = 500
        return status_code, CategoriesResponse(
            message=get_message_by_status(status_code),
            data=None
        )