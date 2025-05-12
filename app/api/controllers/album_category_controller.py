from functools import partial

import torch
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.cache import get_cached_embeddings_parallel
from app.schemas.album_schema import ImageCategoryGroup, ImageRequest
from app.service.category import categorize_images
from app.utils.logging_decorator import log_exception, log_flow


@log_flow
async def categorize_controller(req: ImageRequest, request: Request):
    translated_categories = request.app.state.translated_categories
    text_features = request.app.state.text_features
    
    loop = request.app.state.loop

    image_names = req.images
    embed_load_func = partial(
        get_cached_embeddings_parallel,
        image_names
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

    image_features = torch.stack(image_features)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    task_func = partial(
        categorize_images,
        image_features.cpu(),
        image_names,
        text_features.cpu(),
        translated_categories,
    )
    categorized = await loop.run_in_executor(
        None,
        task_func
    )

    response = [
        ImageCategoryGroup(category=category, images=images)
        for category, images in categorized.items()
    ]

    return {"message": "success", "data": response}
