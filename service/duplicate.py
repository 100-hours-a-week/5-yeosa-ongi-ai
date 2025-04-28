import torch
from sklearn.cluster import DBSCAN
from collections import defaultdict

def find_duplicate_groups(
    image_features: torch.Tensor,
    image_names: list[str],
    eps: float = 0.1,
    min_samples: int = 2
) -> list[list[str]]:
    """
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
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed') \
             .fit_predict(distance_matrix.numpy())

    # 3. Group by cluster label (exclude noise = -1)
    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            groups[label].append(image_names[idx])

    return list(groups.values())

if __name__ == "__main__":
    imagedata = torch.load("image_features.pt", weights_only=True)
    image_names = imagedata['image_names']
    image_features = imagedata['image_features']
    duplicate_image_groups = find_duplicate_groups(image_features, image_names)
    print('done')