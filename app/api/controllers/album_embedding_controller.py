from functools import partial

from fastapi import Request

from app.schemas.album_schema import ImageRequest
from app.service.embedding import embed_images  # service import
from app.utils.logging_decorator import log_exception, log_flow


@log_flow
async def embed_controller(req: ImageRequest, request: Request):
    image_refs = req.images
    image_loader = request.app.state.image_loader

    images = await image_loader.load_images(image_refs)

    clip_model = request.app.state.clip_model
    clip_preprocess = request.app.state.clip_preprocess
    loop = request.app.state.loop
    task_func = partial(
        embed_images,
        clip_model,
        clip_preprocess,
        images,
        image_refs,
        batch_size=16,
        device="cpu"
    )

    await loop.run_in_executor(
        None,
        task_func
    )

    return {"message": "success", "data": None}
