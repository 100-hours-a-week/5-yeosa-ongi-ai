import torch
from collections import Counter, defaultdict

def compute_similarity(
    image_features: torch.Tensor, text_features: torch.Tensor, topk: int
):
    similarity = image_features @ text_features.T  # [B, T]
    topk_scores, topk_indices = similarity.topk(topk, dim=1)
    return topk_scores, topk_indices


def filter_tags_by_threshold(
    image_names, topk_scores, topk_indices, categories, threshold
):
    results = []
    tag_counter = Counter()

    for name, scores, indices in zip(image_names, topk_scores, topk_indices):
        tags = []
        for i, s in zip(indices, scores):
            score = s.item()
            if score >= threshold:
                tag = categories[i]
                tags.append(tag)
                tag_counter[tag] += 1
        results.append((name, tags))
    return results, tag_counter


def assign_primary_category(tags, top_tags):
    for tag in tags:
        if tag in top_tags:
            return tag
    return "기타"


def group_images_by_category(filtered_results, top_tags):
    grouped = defaultdict(list)
    for name, tags in filtered_results:
        category = assign_primary_category(tags, top_tags)
        grouped[category].append(name)
    return dict(grouped)


def categorize_images(
    image_features: torch.Tensor,
    image_names: list[str],
    text_features: torch.Tensor,
    categories: list[str],
    topk: int = 3,
    similarity_threshold: float = 0.25,
):
    topk_scores, topk_indices = compute_similarity(
        image_features, text_features, topk
    )
    filtered_results, tag_counter = filter_tags_by_threshold(
        image_names,
        topk_scores,
        topk_indices,
        categories,
        similarity_threshold,
    )
    top_tags = set(tag for tag, _ in tag_counter.most_common(5))
    return group_images_by_category(filtered_results, top_tags)


if __name__ == "__main__":
    data = torch.load("category_features.pt", weights_only=True)
    categories = data["categories"]
    text_features = data["text_features"]
    imagedata = torch.load("image_features.pt", weights_only=True)
    image_names = imagedata["image_names"]
    image_features = imagedata["image_features"]
    image_features /= image_features.norm(dim=-1, keepdim=True)
    categorize_images(
        image_features.cpu(), image_names, text_features.cpu(), categories
    )
    print("done")
