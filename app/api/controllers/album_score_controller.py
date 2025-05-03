from itertools import chain

import torch
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.cache import get_cached_embeddings_parallel
from app.schemas.album_schema import CategoryScoreRequest
from app.service.highlight import estimate_highlight_score
from app.utils.logging_decorator import log_exception


@log_exception
def highlight_scoring_controller(req: CategoryScoreRequest, request: Request):
    categories = req.categories
    all_images = list(
        chain.from_iterable(category.images for category in categories)
    )

    image_features, missing_keys = get_cached_embeddings_parallel(all_images)
    if missing_keys:
        return JSONResponse(
            status_code=428,
            content={"message": "embedding_required", "data": missing_keys},
        )

    embedding_map = {
        image: feature for image, feature in zip(all_images, image_features)
    }

    data = []
    aesthetic_regressor = request.app.state.aesthetic_regressor

    for category in categories:
        image_features = torch.stack([
            embedding_map[image] for image in category.images
        ])
        image_features /= image_features.norm(dim=-1, keepdim=True)
        scores = estimate_highlight_score(
            image_features, category.images, aesthetic_regressor
        )
        data.append({"category": category.category, "images": scores})

    return JSONResponse(
        status_code=201, content={"message": "success", "data": data}
    )
