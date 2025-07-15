import logging
from typing import Dict, List, Any

import torch

from app.utils.logging_decorator import log_exception, log_flow
from app.schemas.models.score import ScoreCategory, ScoreImage

logger = logging.getLogger(__name__)


@log_flow
def score_each_category(
    categories: List[Any],
    embedding_map: Dict[str, torch.Tensor],
    regressor: torch.nn.Module,
) -> List[ScoreCategory]:
    """
    각 카테고리별로 이미지들의 하이라이트 점수를 계산합니다.

    Args:
        categories: 카테고리 객체 리스트 (각 카테고리는 images 속성을 가짐)
        embedding_map: 이미지 파일명을 키로, 임베딩을 값으로 하는 딕셔너리
        regressor: 하이라이트 점수를 예측하는 회귀 모델

    Returns:
        List[Dict[str, Any]]: 각 카테고리별 이미지 점수 정보
            [
                {
                    "category": str,
                    "images": List[Dict[str, float]]  # [{"image": str, "score": float}, ...]
                },
                ...
            ]

    """
    logger.info(
        "카테고리별 하이라이트 점수 계산 시작",
        extra={"total_categories": len(categories)},
    )

    scored_categories: List[ScoreCategory] = []
    for category in categories:
        logger.debug(
            "카테고리 처리 중",
            extra={"category": category.category, "image_count": len(category.images)},
        )

        image_features = torch.stack([
            embedding_map[image] for image in category.images
        ])
        image_features /= image_features.norm(dim=-1, keepdim=True)

        scores = estimate_highlight_score(
            image_features, category.images, regressor
        )

        scored_category = ScoreCategory(
            category=category.category,
            images=scores
        )

        scored_categories.append(scored_category)

    logger.info(
        "카테고리별 하이라이트 점수 계산 완료",
        extra={"processed_categories": len(scored_categories)},
    )

    return scored_categories


@log_flow
def estimate_highlight_score(
    image_features: torch.Tensor,
    image_names: List[str],
    aesthetic_regressor: torch.nn.Module,
) -> List[ScoreImage]:
    """
    이미지들의 하이라이트 점수를 예측합니다.

    Args:
        image_features: [N, 512] 형태의 이미지 임베딩 텐서
        image_names: 이미지 파일명 리스트 (N개)
        aesthetic_regressor: 이미지당 점수를 출력하는 학습된 회귀 모델

    Returns:
        List[Dict[str, float]]: 각 이미지의 점수 정보
            [{"image": 이미지명, "score": 점수}, ...]

    """
    logger.debug(
        "하이라이트 점수 예측 시작",
        extra={"total_images": len(image_names)},
    )

    aesthetic_regressor.eval()
    with torch.no_grad():
        scores = aesthetic_regressor(image_features)  # shape: [N]

    result = [
        ScoreImage(image=name, score=score.item())
        for name, score in zip(image_names, scores)
    ]

    logger.debug(
        "하이라이트 점수 예측 완료",
        extra={"processed_images": len(result)},
    )

    return result
