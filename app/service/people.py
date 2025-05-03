import numpy as np
import torch
from torch import Tensor
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from app.utils.logging_utils import log_exception
from typing import List

device = "cpu"


def preprocess(face: np.ndarray) -> np.ndarray:
    face = np.transpose(face, (2, 0, 1)).astype(np.float32)
    return (face - 127.5) / 128.0

@log_exception
def cluster_faces(
    images: List, file_names: List[str], arcface_model, yolo_detector
) -> List[List[str]]:
    detector = yolo_detector
    arcface = arcface_model

    np_images = [np.array(img) for img in images]
    bboxes_list, landmarks_list = detector.predict(np_images)

    crops = []
    mapped_names = []

    for idx, (img, bboxes, landmarks) in enumerate(
        zip(np_images, bboxes_list, landmarks_list)
    ):
        if len(landmarks) > 0:
            aligned = detector.align(img, landmarks)
            for face in aligned:
                crops.append(face)
                mapped_names.append(file_names[idx])

    if not crops:
        return []

    input_tensor: Tensor = torch.tensor(
        np.stack([preprocess(f) for f in crops]),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        embeddings = arcface(input_tensor).cpu().numpy()

    dbscan = DBSCAN(eps=0.6, min_samples=2, metric="cosine")
    labels = dbscan.fit_predict(embeddings)

    # 클러스터별 평균 거리 기반 필터링
    cluster_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            cluster_indices[label].append(idx)

    valid_labels = []
    for label, idxs in cluster_indices.items():
        cluster_embeds = embeddings[idxs]
        n = len(cluster_embeds)
        if n < 2:
            continue
        dists = cosine_distances(cluster_embeds)
        avg_dist = np.mean(dists[np.triu_indices(n, k=1)])
        max_dist = np.max(dists[np.triu_indices(n, k=1)])
        
        if n < 5:
            if max_dist < 0.6:
                valid_labels.append(label)
        else:
            if avg_dist < 0.5:
                valid_labels.append(label)

    # 유효 클러스터만 반영한 라벨로 재구성
    filtered_labels = np.array([
        label if label in valid_labels else -1
        for label in labels
    ])

    # 결과 정리
    result = defaultdict(set)
    for label, name in zip(filtered_labels, mapped_names):
        if label != -1:
            result[f"person_{label}"].add(name)

    # 반환: 리스트로 변환
    sorted_result = sorted(result.values(), key=lambda names: len(names), reverse=True)
    return [list(names) for names in sorted_result]
