import torch
import torch.nn.functional as F
from typing import List, Dict

def compute_pairwise_score(image_features, text_pair):
    """
    이미지 임베딩과 두 개의 텍스트 임베딩 쌍(positive, negative)에 대해
    softmax를 통해 positive 점수를 계산합니다.

    인자:
        image_features (Tensor): [B, 512] 이미지 피처 (정규화되어 있다고 가정)
        text_pair (Tensor): [2, 512] 텍스트 쌍 (positive, negative, 정규화되어 있다고 가정)

    반환:
        Tensor: [B] 각 이미지에 대한 positive softmax 점수
    """
    sim = image_features @ text_pair.T         # 코사인 유사도 [B, 2]
    probs = F.softmax(sim, dim=-1)             # softmax 적용
    return probs[:, 0]                         # positive 클래스의 확률 반환


def get_field_scores(image_features: torch.Tensor, text_features: torch.Tensor, fields: List[str]) -> List[Dict[str, float]]:
    """
    각 이미지에 대해 모든 필드의 positive 점수를 계산합니다.

    인자:
        image_features (Tensor): [B, 512] (정규화된 이미지 임베딩, B: Batch_size)
        text_features (Tensor): [N, 2, 512] (정규화된 텍스트 쌍, N: Scoring 하고자 하는 Field 개수(Sharp, Contrast 등))
        fields (List[str]): 각 필드 이름

    반환:
        List[Dict[str, float]]: 각 이미지별 필드 이름 → 점수 딕셔너리
    """
    B, N = image_features.size(0), text_features.size(0)
    scores = [dict() for _ in range(B)]  # 이미지별 점수를 저장할 딕셔너리 리스트

    for i in range(N):  # 필드별로
        pos_scores = compute_pairwise_score(image_features, text_features[i])  # [B]
        for b in range(B):
            scores[b][fields[i]] = pos_scores[b].item()  # field 이름 → 점수로 저장

    return scores

def load_clip_iqa_prompt_features(path: str):
    """
    저장된 .pt 파일에서 CLIP-IQA 프롬프트 정보를 불러옵니다.

    인자:
        path (str): 프롬프트 피처가 저장된 파일 경로 (예: 'clip_iqa_prompt_features.pt')

    반환:
        Tuple:
            prompt_pairs (List[Tuple[str, str]]): positive/negative 텍스트 프롬프트 쌍
            text_features (Tensor): [N, 2, 512] 텍스트 피처 쌍
            fields (List[str]): 각 프롬프트 쌍에 해당하는 필드 이름
    """
    data = torch.load(path)
    return data["text_features"], data["fields"]


def evaluate_dual_threshold(scores, field_a, field_b,
                             weight_b=0.25, threshold_combined=0.490, threshold_a=0.488):
    """
    두 가지 기준을 바탕으로 각 이미지의 통과 여부를 판별합니다:
    - field_a는 단일 기준을 통과해야 함 (ex. good ≥ 0.488)
    - field_a와 field_b의 가중 평균은 통합 기준을 통과해야 함 (ex. good+sharp ≥ 0.490)

    인자:
        scores (List[Dict[str, float]]): 각 이미지의 필드별 점수
        field_a (str): 단일 기준 필드 이름 (예: 'good')
        field_b (str): 보조 필드 이름 (예: 'sharp')
        weight_b (float): field_b에 부여할 가중치 (나머지는 field_a)
        threshold_combined (float): 가중 평균에 적용할 통합 기준
        threshold_a (float): field_a 단일 기준 점수

    반환:
        List[str]: 이미지별 판별 결과 ('both', 'field_a_only', 'combined_only', 'neither')
    """
    results = []
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

    return results

def get_low_quality_images(image_names, image_features):
    """
    'both'가 아닌 모든 결과를 저품질로 간주하고 해당 이미지 이름을 반환합니다.

    인자:
        image_names (List[str]): 이미지 이름 리스트 (B 길이)
        pass_results (List[str]): 평가 결과 ('both', 'field_a_only', 'combined_only', 'neither')

    반환:
        List[str]: 저품질 이미지 이름 리스트
    """
    text_features, fields = load_clip_iqa_prompt_features('app/model/clip_iqa_prompt_features.pt')
    scores = get_field_scores(image_features, text_features, fields)
    results = evaluate_dual_threshold(
        scores,
        field_a="sharp",
        field_b="good",
        weight_b=0.25,
        threshold_combined=0.490,
        threshold_a=0.488
    )
    return [name for name, result in zip(image_names, results) if result != "both"]
