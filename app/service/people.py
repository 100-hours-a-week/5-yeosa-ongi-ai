import numpy as np
import torch
from torch import Tensor
from collections import defaultdict
from sklearn.cluster import DBSCAN
from typing import List, Dict

device = "cpu"


def preprocess(face: np.ndarray) -> np.ndarray:
    face = np.transpose(face, (2, 0, 1)).astype(np.float32)
    return (face - 127.5) / 128.0


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

    result = defaultdict(set)

    for label, name in zip(labels, mapped_names):
        if label != -1:
            result[f"person_{label}"].add(name)

    return [list(names) for names in result.values()]
