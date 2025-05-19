from collections import defaultdict

import torch
from sklearn.cluster import DBSCAN

from app.utils.logging_decorator import log_exception, log_flow

def compute_hashes(images: List[np.ndarray]) -> np.ndarray:
    """
    입력 이미지 리스트에 대해 OpenCV pHash를 계산합니다.

    Args:
        images (List[np.ndarray]): OpenCV 이미지 배열 리스트

    Returns:
        np.ndarray: shape (N, 64) pHash 결과
    """
    hasher = cv2.img_hash.PHash_create()
    return np.vstack([hasher.compute(img) for img in images])


def compute_hamming_matrix(hashes: np.ndarray) -> np.ndarray:
    """
    이미지 해시 간 해밍 거리 행렬을 계산합니다.

    Args:
        hashes (np.ndarray): shape (N, 64) 해시 배열

    Returns:
        np.ndarray: shape (N, N) 해밍 거리 행렬
    """
    expanded = np.expand_dims(hashes, axis=1)
    xor_matrix = np.bitwise_xor(expanded, hashes)
    hamming_matrix = np.unpackbits(xor_matrix, axis=-1).sum(axis=-1)
    return hamming_matrix.astype(np.uint8)


def cluster_with_dbscan(
    hamming_matrix: np.ndarray, eps: int = 10, min_samples: int = 2
) -> np.ndarray:
    """
    해밍 거리 기반 DBSCAN 클러스터링을 수행합니다.

    Args:
        hamming_matrix (np.ndarray): 사전 계산된 거리 행렬 (N, N)
        eps (int, optional): 최대 해밍 거리 임계값. Defaults to 10.
        min_samples (int, optional): 최소 클러스터 크기. Defaults to 2.

    Returns:
        np.ndarray: shape (N,) 클러스터 레이블 (-1은 노이즈)
    """
    clustering = DBSCAN(
        metric="precomputed", eps=eps, min_samples=min_samples
    )
    return clustering.fit_predict(hamming_matrix)


def group_by_labels(labels: np.ndarray, file_names: List[str]) -> List[List[str]]:
    """
    DBSCAN 레이블을 기반으로 파일명을 클러스터별로 그룹화합니다.

    Args:
        labels (np.ndarray): DBSCAN 결과 레이블 (N,)
        file_names (List[str]): 파일명 리스트

    Returns:
        List[List[str]]: 유사 이미지 그룹 목록
    """
    groups = {}

    for idx, label in enumerate(labels):
        if label == -1:
            continue  # 노이즈 제외
        groups.setdefault(label, []).append(file_names[idx])

    return list(groups.values())

@log_flow
def find_duplicate_groups(
    image_features: torch.Tensor,
    image_names: list[str],
    eps: float = 0.1,
    min_samples: int = 2,
) -> list[list[str]]:
    """
    중복 이미지를 클러스터링합니다.

    Args:
        image_features: [N, D] torch.Tensor - 이미지 임베딩 (CPU)
        image_names: list of N strings
        eps: cosine distance threshold
        min_samples: minimum samples to form a cluster

    Returns:
        List of lists of image names that are similar

    """
    # 1. Normalize → Cosine similarity → Cosine distance
    normed = image_features / image_features.norm(dim=1, keepdim=True)
    similarity_matrix = normed @ normed.T
    distance_matrix = 1 - similarity_matrix  # cosine distance
    distance_matrix = distance_matrix.clamp(min=0)

    # 2. DBSCAN clustering (precomputed distance)
    labels = DBSCAN(
        eps=eps, min_samples=min_samples, metric="precomputed"
    ).fit_predict(distance_matrix.numpy())

    # 3. Group by cluster label (exclude noise = -1)
    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            groups[label].append(image_names[idx])

    return list(groups.values())
