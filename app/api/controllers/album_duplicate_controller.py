from functools import partial

import torch
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.cache import get_cached_embeddings_parallel
from app.schemas.album_schema import ImageRequest
from app.service.duplicate import find_duplicate_groups
from app.utils.logging_decorator import log_exception


@log_exception
async def duplicate_controller(req: ImageRequest, request: Request):
    loop = request.app.state.loop

    image_names = req.images
    image_load_func = partial(
        get_cached_embeddings_parallel,
        image_names
    )
    image_features, missing_keys = await loop.run_in_executor(
        None,
        image_load_func
    )
    if missing_keys:
        return JSONResponse(
            status_code=428,
            content={"message": "embedding_required", "data": missing_keys},
        )
        

    image_features = torch.stack(image_features)
    
    task_func = partial(find_duplicate_groups, image_features, image_names)
    data = await loop.run_in_executor(None, task_func)
    return {"message": "success", "data": data}
