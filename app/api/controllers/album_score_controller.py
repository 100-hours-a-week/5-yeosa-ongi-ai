from functools import partial
from itertools import chain

import torch
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.cache import get_cached_embeddings_parallel
from app.schemas.album_schema import CategoryScoreRequest
from app.service.highlight import score_each_category
from app.utils.logging_decorator import log_exception, log_flow


@log_flow
async def highlight_scoring_controller(req: CategoryScoreRequest, request: Request):
    categories = req.categories
    all_images = list(
        chain.from_iterable(category.images for category in categories)
    )

    loop = request.app.state.loop
    
    embed_load_func = partial(
        get_cached_embeddings_parallel,
        all_images
    )
    image_features, missing_keys = await loop.run_in_executor(
        None,
        embed_load_func
    )
    if missing_keys:
        return JSONResponse(
            status_code=428,
            content={"message": "embedding_required", "data": missing_keys},
        )

    embedding_map = {
        image: feature for image, feature in zip(all_images, image_features)
    }

    aesthetic_regressor = request.app.state.aesthetic_regressor
    task_func = partial(score_each_category, categories, embedding_map, aesthetic_regressor)
    data = await loop.run_in_executor(None, task_func)

    return JSONResponse(
        status_code=201, content={"message": "success", "data": data}
    )
