from app.utils.image_loader import get_image_loader
from app.service.embedding import embed_images  # service import
from app.schemas.album_schema import ImageRequest
from fastapi import Request


def embed_controller(req: ImageRequest, request: Request):
    filenames = req.images

    image_loader = get_image_loader()
    images = image_loader.load_images(filenames)

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
