import torch
from app.core.cache import get_cached_embedding, get_cached_embeddings_parallel
from fastapi.responses import JSONResponse
from app.schemas.album_schema import ImageRequest
from app.service.duplicate import find_duplicate_groups


def duplicate_controller(req: ImageRequest):
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
    image_features /= image_features
