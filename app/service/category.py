from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch

from app.utils.logging_decorator import log_exception, log_flow


def compute_similarity(
    image_features: torch.Tensor, 
    text_features: torch.Tensor, 
) -> torch.Tensor:
    """
    이미지와 텍스트 특징 간의 유사도를 계산합니다.

    Args:
        image_features: 이미지 특징 텐서 [N, 512]
        text_features: 텍스트 특징 텐서 [T, 4, 512]

    Returns:
        sims_matrix: 이미지-태그 간 유사도 행렬 [N, T]
    """
    # einsum: [N, 512(d)] · [T, 4(p), 512(d)]ᵀ → [N, T, 4(p)]
    sims_per_prompt = torch.einsum("nd,tpd->ntp", image_features, text_features)
    
    # 태그별 유사도를 프롬프트별 유사도의 평균으로 선택 → [N, T]
    sims_matrix = sims_per_prompt.mean(dim=-1)
    
    return sims_matrix


def apply_tag_boosts(
    sims_matrix: torch.Tensor,
    categories: List[str],
    tag_boosts: Dict[str, float],
    threshold: float = 0.22
) -> torch.Tensor:
    """
    특정 태그에 대해 유사도 점수를 보정합니다.

    Args:
        sims_matrix: 유사도 행렬 [N, T]
        categories: 카테고리 리스트
        tag_boosts: 태그별 보정 계수
        threshold: 보정을 적용할 임계값

    Returns:
        보정된 유사도 행렬
    """
    for i, tag in enumerate(categories):
        if tag in tag_boosts:
            mask = sims_matrix[:, i] <= threshold
            sims_matrix[mask, i] *= tag_boosts[tag]
    
    return sims_matrix


def select_topk_tags_per_image(
    sims_matrix: torch.Tensor,
    categories: List[str],
    k: int = 3
) -> List[List[Tuple[str, float]]]:
    """
    각 이미지별로 상위 k개의 태그를 선택합니다.

    Args:
        sims_matrix: 유사도 행렬 [N, T]
        categories: 카테고리 리스트
        k: 각 이미지당 선택할 태그 수

    Returns:
        각 이미지별 topk 태그 정보 [(태그, 점수), ...]
    """
    topk_scores, topk_indices = torch.topk(sims_matrix, k=k, dim=1)
    topk_info = [
        [(categories[idx], score.item()) for idx, score in zip(indices, scores)]
        for indices, scores in zip(topk_indices, topk_scores)
    ]
    
    return topk_info


def compute_tag_representative_scores(
    topk_info: List[List[Tuple[str, float]]],
    categories: List[str],
    tau: float = 0.28,
    lambda_boost: float = 0.5
) -> List[Tuple[str, float]]:
    """
    각 태그의 대표성 점수를 계산합니다.

    Args:
        topk_info: 각 이미지별 top3 태그 정보 [(태그, 점수), ...]
        categories: 카테고리 리스트
        tau: 신뢰도 임계값
        lambda_boost: 부스트 가중치

    Returns:
        각 태그의 대표성 점수 [(태그, 점수), ...]
    """
    sum_scores = torch.zeros(len(categories))
    bonus_scores = torch.zeros(len(categories))
    
    for image_tags in topk_info:
        for tag, score in image_tags:
            tag_idx = categories.index(tag)
            sum_scores[tag_idx] += score
            if score > tau:
                bonus_scores[tag_idx] += score
    
    tag_representative_scores = sum_scores + lambda_boost * bonus_scores
    return [(categories[i], score.item()) for i, score in enumerate(tag_representative_scores)]


def select_representative_categories(
    tag_representative_scores: List[Tuple[str, float]],
    k: int = 5
) -> List[Tuple[str, float]]:
    """
    대표성 점수가 가장 높은 상위 k개 태그를 선택합니다.

    Args:
        tag_representative_scores: 각 태그의 대표성 점수 [(태그, 점수), ...]
        k: 선택할 태그 수

    Returns:
        상위 k개 태그와 점수 [(태그, 점수), ...]

    """
    # 점수 기준으로 정렬하여 상위 k개 선택
    return sorted(tag_representative_scores, key=lambda x: x[1], reverse=True)[:k]


def classify_images_by_representative_tags(
    topk_info: List[List[Tuple[str, float]]],
    representative_tags: List[Tuple[str, float]],
    threshold: float = 0.21
) -> Dict[str, List[int]]:
    """
    1차 이미지 분류를 수행합니다.

    Args:
        topk_info: 각 이미지별 top3 태그 정보 [(태그, 점수), ...]
        representative_tags: 대표 태그 리스트 [(태그, 점수), ...]
        threshold: 유사도 임계값

    Returns:
        카테고리별 이미지 정보
        - key: 카테고리 이름
        - value: [이미지_인덱스, ...]
    """
    rep_tags = {tag for tag, _ in representative_tags}
    
    category_to_images = defaultdict(list)
    
    for i, image_tags in enumerate(topk_info):
        # top3 태그를 순서대로 확인
        for tag, score in image_tags:
            if tag in rep_tags and score >= threshold:
                category_to_images[tag].append(i)
                break
        else:  # 모든 태그가 조건을 만족하지 않으면
            category_to_images["기타"].append(i)
    
    return dict(category_to_images)


def select_representative_tag_per_category(
    category_to_images: Dict[str, List[int]],
    topk_info: List[List[Tuple[str, float]]],
    categories: List[str],
    tau: float = 0.28,
    lambda_boost: float = 0.5
) -> Dict[str, Tuple[str, float]]:
    """
    각 카테고리별로 대표 태그를 선정합니다.

    Args:
        category_to_images: 카테고리별 이미지 인덱스 {카테고리: [이미지_인덱스, ...]}
        topk_info: 각 이미지별 top3 태그 정보 [(태그, 점수), ...]
        tau: 신뢰도 임계값
        lambda_boost: 부스트 가중치

    Returns:
        카테고리별 대표 태그 정보 {카테고리: (대표_태그, 점수)}
    """
    category_to_rep_tag = {}
    
    for category, image_indices in category_to_images.items():
        if category == "기타":
            continue
            
        category_topk_info = [topk_info[i] for i in image_indices]
        
        tag_representetiva_scores = compute_tag_representative_scores(
            category_topk_info,
            categories,
            tau,
            lambda_boost
        )
        
        rep_tag = select_representative_categories(tag_representetiva_scores, k=1)[0]
        
        category_to_rep_tag[category] = rep_tag
    
    return category_to_rep_tag


def reclassify_images_by_new_rep_tags(
    category_to_images: Dict[str, List[int]],
    category_to_rep_tag: Dict[str, Tuple[str, float]],
    topk_info: List[List[Tuple[str, float]]],
    threshold: float = 0.21
) -> Dict[str, List[int]]:
    """
    새로운 대표 태그를 기준으로 이미지를 재분류합니다.

    Args:
        category_to_images: 카테고리별 이미지 인덱스 {카테고리: [이미지_인덱스, ...]}
        category_to_rep_tag: 카테고리별 새로운 대표 태그 {카테고리: (대표_태그, 점수)}
        topk_info: 각 이미지별 top3 태그 정보 [(태그, 점수), ...]
        threshold: 유사도 임계값

    Returns:
        재분류된 카테고리별 이미지 인덱스
    """
    new_category_to_images = defaultdict(list)
    
    for category, image_indices in category_to_images.items():
        if category == "기타":
            # 기타 카테고리는 그대로 유지
            new_category_to_images["기타"].extend(image_indices)
            continue
            
        new_rep_tag, _ = category_to_rep_tag[category]
        
        # 기존 카테고리와 새로운 대표 태그가 같으면 재분류하지 않음
        if category == new_rep_tag:
            new_category_to_images[category].extend(image_indices)
            continue
        
        # 새로운 대표 태그로 재분류
        for img_idx in image_indices:
            # top3 태그 중에 새로운 대표 태그가 있는지 확인
            found = False
            for tag, score in topk_info[img_idx]:
                if tag == new_rep_tag and score >= threshold:
                    new_category_to_images[new_rep_tag].append(img_idx)
                    found = True
                    break
            
            # 새로운 대표 태그가 없거나 임계값 미만이면 기타로
            if not found:
                new_category_to_images["기타"].append(img_idx)
    
    return dict(new_category_to_images)


@log_flow
def categorize_images(
    image_features: torch.Tensor,
    image_names: List[str],  # 이미지 이름 리스트 추가
    text_features: torch.Tensor,
    categories: List[str],
    tag_boosts: Optional[Dict[str, float]] = None,
    tau: float = 0.28,
    lambda_boost: float = 0.5,
    threshold: float = 0.21
) -> Dict[str, List[str]]:  # 반환 타입을 Dict[str, List[str]]로 변경
    """
    이미지들을 카테고리별로 분류합니다.

    Args:
        image_features: 이미지 특징 벡터 [N, D]
        image_names: 이미지 이름 리스트 [N]
        text_features: 태그 특징 벡터 [T, 4, D]
        categories: 카테고리 리스트
        tag_boosts: 태그별 보정 계수 (기본값: None)
        tau: 신뢰도 임계값
        lambda_boost: 부스트 가중치
        threshold: 유사도 임계값

    Returns:
        카테고리별 이미지 이름 {카테고리: [이미지_이름, ...]}
    """
    # 1. 이미지-태그 유사도 계산 및 보정
    sims_matrix = compute_similarity(image_features, text_features)
    boosted_sims_matrix = apply_tag_boosts(sims_matrix, categories, tag_boosts, threshold) if tag_boosts else sims_matrix
    
    # 2. 이미지별 top3 태그 추출
    topk_info = select_topk_tags_per_image(boosted_sims_matrix, categories)
    
    # 3. 대표 태그 선정
    tag_scores = compute_tag_representative_scores(topk_info, categories, tau, lambda_boost)
    representative_tags = select_representative_categories(tag_scores)
    
    # 4. 1차 분류
    category_to_images = classify_images_by_representative_tags(
        topk_info, representative_tags, threshold
    )
    
    # 5. 각 카테고리별 새로운 대표 태그 선정
    category_to_rep_tag = select_representative_tag_per_category(
        category_to_images, topk_info, categories, tau, lambda_boost
    )
    
    # 6. 새로운 대표 태그로 재분류
    final_category_to_indices = reclassify_images_by_new_rep_tags(
        category_to_images, category_to_rep_tag, topk_info, threshold
    )
    
    # 인덱스를 이미지 이름으로 변환
    return {
        category: [image_names[idx] for idx in indices]
        for category, indices in final_category_to_indices.items()
    }