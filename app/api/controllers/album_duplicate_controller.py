import torch
from fastapi.responses import JSONResponse

from app.core.cache import get_cached_embeddings_parallel
from app.schemas.album_schema import ImageRequest
from app.service.duplicate import find_duplicate_groups
from app.utils.logging_decorator import log_exception


@log_exception
def duplicate_controller(req: ImageRequest):
    image_names = req.images

    image_features, missing_keys = get_cached_embeddings_parallel(image_names)
    if missing_keys:
        return JSONResponse(
            status_code=428,
            content={"message": "embedding_required", "data": missing_keys},
        )

    image_features = torch.stack(image_features)
    data = find_duplicate_groups(image_features, image_names)
    return {"message": "success", "data": data}
