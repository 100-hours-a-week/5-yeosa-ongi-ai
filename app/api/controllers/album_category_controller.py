import torch
from fastapi.responses import JSONResponse
from app.core.cache import get_cached_embedding, get_cached_embeddings_parallel
from app.schemas.album_schema import ImageRequest, ImageCategoryGroup
from app.service.category import categorize_images
from fastapi import Request


def categorize_controller(req: ImageRequest, request: Request):
    data = torch.load("app/model/category_features.pt", weights_only=True)
    translated_categories = data["translated_categories"]
    text_features = data["text_features"]

    image_names = req.images

    image_features, missing_keys = get_cached_embeddings_parallel(image_names)
    if missing_keys:
        return JSONResponse(
            status_code=428,
            content={"message": "embedding_required", "data": missing_keys},
        )

    image_features = torch.stack([
        get_cached_embedding(image_name) for image_name in image_names
    ])
    image_features /= image_features.norm(dim=-1, keepdim=True)

    categorized = categorize_images(
        image_features.cpu(), image_names, text_features.cpu(), translated_categories
    )

    response = [
        ImageCategoryGroup(category=category, images=images)
        for category, images in categorized.items()
    ]

    return {"message": "success", "data": response}
