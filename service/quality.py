import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

if __name__ == "__main__":
    
    text_features = torch.randn(10, 512)
    image_features = torch.randn(100, 512)

    image_features_np = image_features.cpu().numpy()
    similarity = cosine_similarity(image_features_np, text_features)

    num_criteria = 5
    quality_scores = np.stack([
    similarity[:, i * 2] - similarity[:, i * 2 + 1]
    for i in range(num_criteria)
    ], axis=1)  # shape: [N, 5]

    threshold = 0.2  # 낮은 점수면 저품질로 간주
    low_quality_mask = (quality_scores < threshold).any(axis=1)