import torch
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.cache import get_cached_embeddings_parallel
from app.schemas.album_schema import ImageRequest
from app.service.quality import get_low_quality_images
from app.utils.logging_decorator import log_exception


@log_exception
def quality_controller(req: ImageRequest, request: Request):
    image_names = req.images

    image_features, missing_keys = get_cached_embeddings_parallel(image_names)
    if missing_keys:
        return JSONResponse(
            status_code=428,
            content={"message": "embedding_required", "data": missing_keys},
        )

    image_features = torch.stack(image_features)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    result = get_low_quality_images(image_names, image_features)

    return {"message": "success", "data": result}
