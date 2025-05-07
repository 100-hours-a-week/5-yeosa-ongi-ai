import torch

from app.utils.logging_decorator import log_exception, log_flow

@log_flow
def score_each_category(categories, embedding_map, regressor):
    data = []
    for category in categories:
        image_features = torch.stack([
            embedding_map[image] for image in category.images
        ])
        image_features /= image_features.norm(dim=-1, keepdim=True)
        scores = estimate_highlight_score(
            image_features, category.images, regressor
        )
        data.append({"category": category.category, "images": scores})
    return data

@log_flow
def estimate_highlight_score(
    image_features: torch.Tensor,
    image_names: list[str],
    aesthetic_regressor: torch.nn.Module,
) -> list[dict[str, float]]:
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
