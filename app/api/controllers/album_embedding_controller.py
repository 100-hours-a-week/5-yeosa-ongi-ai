from fastapi import Request

from app.schemas.album_schema import ImageRequest
from app.service.embedding import embed_images  # service import
from app.utils.logging_decorator import log_exception


@log_exception
async def embed_controller(req: ImageRequest, request: Request):
    filenames = req.images

    image_loader = request.app.state.image_loader
    images = await image_loader.load_images(filenames)

    clip_model = request.app.state.clip_model
    clip_preprocess = request.app.state.clip_preprocess

    embed_images(
        clip_model,
        clip_preprocess,
        images,
        filenames,
        batch_size=16,
        device="cpu",
    )

    return {"message": "success", "data": None}
