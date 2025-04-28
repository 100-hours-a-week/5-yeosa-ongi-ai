import torch
from typing import List, Dict

def estimate_highlight_score(
    image_features: torch.Tensor,
    image_names: List[str],
    aesthetic_regressor: torch.nn.Module
) -> List[Dict[str, float]]:
    """
    Estimate highlight scores for images using an aesthetic regressor model.

    Args:
        image_features (torch.Tensor): Tensor of shape [N, 512] representing image embeddings.
        image_names (List[str]): List of N image file names corresponding to image_features.
        aesthetic_regressor (torch.nn.Module): Trained regression model that outputs a score per image.

    Returns:
        List[Dict[str, float]]: Each element contains {'image': image_name, 'score': score}.
    """
    aesthetic_regressor.eval()
    with torch.no_grad():
        scores = aesthetic_regressor(image_features)  # shape: [N]

    return [
        {"image": name, "score": score.item()}
        for name, score in zip(image_names, scores)
    ]