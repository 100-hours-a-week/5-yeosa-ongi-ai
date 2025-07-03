import logging
from typing import Dict, List, Literal, Tuple

import torch
import torch.nn.functional as F
import cv2
import numpy as np

from app.core.cache import get_cached_embeddings_parallel
from app.utils.logging_decorator import log_exception, log_flow
from app.config.settings import MODEL_NAME

logger = logging.getLogger(__name__)

# 상수 정의
DEFAULT_WEIGHT_B = 0.25
DEFAULT_THRESHOLD_COMBINED = 0.486 if MODEL_NAME.value == 'ViT-L/14' else 0.490
DEFAULT_THRESHOLD_A = 0.483 if MODEL_NAME.value == 'ViT-L/14' else 0.488

ResultType = Literal["both", "field_a_only", "combined_only", "neither"]


@log_flow
def compute_pairwise_score(
    image_features: torch.Tensor,
    text_pair: torch.Tensor,
) -> torch.Tensor:
    """
    이미지 임베딩과 두 개의 텍스트 임베딩 쌍(positive, negative)에 대해 softmax를 통해 positive 점수를 계산합니다.

    Args:
        image_features: [B, 512] 이미지 피처 (정규화되어 있다고 가정)
        text_pair: [2, 512] 텍스트 쌍 (positive, negative, 정규화되어 있다고 가정)

    Returns:
        torch.Tensor: [B] 각 이미지에 대한 positive softmax 점수

    """
    logger.debug(
        "pairwise 점수 계산 시작",
        extra={"batch_size": image_features.size(0)},
    )

    sim = image_features @ text_pair.T  # 코사인 유사도 [B, 2]
    probs = F.softmax(sim, dim=-1)  # softmax 적용
    result = probs[:, 0]  # positive 클래스의 확률 반환

    logger.debug(
        "pairwise 점수 계산 완료",
        extra={"batch_size": len(result)},
    )

    return result


@log_flow
def get_field_scores(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    fields: List[str],
) -> List[Dict[str, float]]:
    """
    각 이미지에 대해 모든 필드의 positive 점수를 계산합니다.

    Args:
        image_features: [B, 512] 정규화된 이미지 임베딩 (B: Batch_size)
        text_features: [N, 2, 512] 정규화된 텍스트 쌍 (N: Scoring 하고자 하는 Field 개수)
        fields: 각 필드 이름 리스트

    Returns:
        List[Dict[str, float]]: 각 이미지별 필드 이름 → 점수 딕셔너리

    """
    logger.info(
        "필드 점수 계산 시작",
        extra={
            "batch_size": image_features.size(0),
            "field_count": len(fields),
        },
    )

    B, N = image_features.size(0), text_features.size(0)
    scores = [dict() for _ in range(B)]  # 이미지별 점수를 저장할 딕셔너리 리스트

    for i in range(N):  # 필드별로
        logger.debug(
            "필드 처리 중",
            extra={"field": fields[i], "progress": f"{i+1}/{N}"},
        )

        pos_scores = compute_pairwise_score(image_features, text_features[i])  # [B]
        for b in range(B):
            scores[b][fields[i]] = pos_scores[b].item()  # field 이름 → 점수로 저장

    logger.info(
        "필드 점수 계산 완료",
        extra={"processed_fields": len(fields)},
    )

    return scores


@log_exception
def evaluate_dual_threshold(
    scores: List[Dict[str, float]],
    field_a: str,
    field_b: str,
    weight_b: float = DEFAULT_WEIGHT_B,
    threshold_combined: float = DEFAULT_THRESHOLD_COMBINED,
    threshold_a: float = DEFAULT_THRESHOLD_A,
) -> List[ResultType]:
    """
    두 가지 기준을 바탕으로 각 이미지의 통과 여부를 판별합니다:

    - field_a는 단일 기준을 통과해야 함 (ex. good ≥ 0.488)
    - field_a와 field_b의 가중 평균은 통합 기준을 통과해야 함 (ex. good+sharp ≥ 0.490)

    Args:
        scores: 각 이미지의 필드별 점수
        field_a: 단일 기준 필드 이름 (예: 'good')
        field_b: 보조 필드 이름 (예: 'sharp')
        weight_b: field_b에 부여할 가중치 (기본값: 0.25)
        threshold_combined: 가중 평균에 적용할 통합 기준 (기본값: 0.490)
        threshold_a: field_a 단일 기준 점수 (기본값: 0.488)

    Returns:
        List[ResultType]: 이미지별 판별 결과

    """
    logger.info(
        "이중 임계값 평가 시작",
        extra={
            "field_a": field_a,
            "field_b": field_b,
            "weight_b": weight_b,
            "threshold_combined": threshold_combined,
            "threshold_a": threshold_a,
        },
    )

    results: List[ResultType] = []
    for s in scores:
        score_a = s[field_a]  # 단일 기준 필드
        score_b = s[field_b]  # 보조 필드
        combined = (1 - weight_b) * score_a + weight_b * score_b

        passed_a = score_a >= threshold_a
        passed_combined = combined >= threshold_combined

        if passed_a and passed_combined:
            results.append("both")
        elif passed_a:
            results.append("field_a_only")
        elif passed_combined:
            results.append("combined_only")
        else:
            results.append("neither")

    logger.info(
        "이중 임계값 평가 완료",
        extra={
            "total_images": len(results),
            "passed_both": results.count("both"),
            "passed_a_only": results.count("field_a_only"),
            "passed_combined_only": results.count("combined_only"),
            "failed": results.count("neither"),
        },
    )

    return results


@log_exception
async def get_clip_low_quality_images(
    image_refs: List[str],
    text_features: torch.Tensor,
    fields: List[str],
) -> Tuple[List[str], List[str]]:
    """
    'both'가 아닌 모든 결과를 저품질로 간주하고 해당 이미지 이름을 반환합니다.

    Args:
        image_refs: 이미지 이름 리스트
        text_features: 텍스트 임베딩 텐서
        fields: 필드 이름 리스트

    Returns:
        Tuple[List[str], List[str]]: 저품질 이미지 이름 리스트, 임베딩이 필요한 키 리스트

    """
    logger.info(
        "저품질 이미지 검색 시작",
        extra={"total_images": len(image_refs)},
    )
    
    
    # 1. 이미지 임베딩 로드

    print("quality 임베딩 로드 전")
    image_features, missing_keys = await get_cached_embeddings_parallel(image_refs)
    print("quality 임베딩 로드 후")

    # 2. 임베딩이 없는 이미지 처리
    if missing_keys:
        logger.warning(
            "일부 이미지의 임베딩이 없음",
            extra={"missing_count": len(missing_keys)},
        )
        return image_features, missing_keys

    # 3. 이미지 임베딩 정규화
    image_features = torch.stack(image_features)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    scores = get_field_scores(image_features, text_features, fields)
    results = evaluate_dual_threshold(
        scores,
        field_a="sharp",
        field_b="good",
        weight_b=DEFAULT_WEIGHT_B,
        threshold_combined=DEFAULT_THRESHOLD_COMBINED,
        threshold_a=DEFAULT_THRESHOLD_A,
    )

    low_quality_images = [
        name for name, result in zip(image_refs, results) if result != "both"
    ]

    logger.info(
        "저품질 이미지 검색 완료",
        extra={
            "total_images": len(image_refs),
            "low_quality_count": len(low_quality_images),
        },
    )

    return low_quality_images, missing_keys


@log_exception
def resize_for_laplacian(image: np.ndarray, target_long_side: int = 300):
    """
    긴 변을 기준으로 이미지 크기를 축소하여 Laplacian 분석용으로 리사이즈합니다.

    Args:
        image (np.ndarray): Grayscale 이미지
        target_long_side (int): 기준 긴 변 픽셀 수 (default: 300)

    Returns:
        np.ndarray: 리사이즈된 Grayscale 이미지
    """
    h, w = image.shape
    scale = target_long_side / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized


@log_exception
def laplacian_filter(
    image: np.ndarray,
    threshold: float = 80.0,
    target_long_side: int = 300,
) -> bool:
    """
    Laplacian을 사용하여 이미지의 품질을 평가합니다.

    Args:
        image (np.ndarray): Grayscale 이미지
        threshold (float): Laplacian 임계값 (default: 80.0)
        target_long_side (int): 기준 긴 변 픽셀 수 (default: 300)

    Returns:
        bool: 품질이 낮으면 True, 그렇지 않으면 False
    """
    resized_image = resize_for_laplacian(image, target_long_side)
    laplacian_var = cv2.Laplacian(resized_image, cv2.CV_64F).var()
    return laplacian_var < threshold

@log_exception
async def get_laplacian_low_quality_images(image_refs: List[str], image_loader, threshold: float = 80.0) -> List[str]:
    """
    이미지 목록에서 Laplacian 필터를 사용하여 저품질 이미지를 검색합니다.

    Args:
        image_refs (List[str]): 이미지 파일명 목록
        image_loader: 이미지 로더 객체
        threshold (float): Laplacian 임계값 (default: 80.0)

    Returns:
        List[str]: 저품질 이미지 파일명 목록
    """
    images = await image_loader.load_images(image_refs, 'GRAY')
    laplacian_low_quality_images = [image_ref for image_ref, image in zip(image_refs, images) if laplacian_filter(image, threshold)]

    return laplacian_low_quality_images
