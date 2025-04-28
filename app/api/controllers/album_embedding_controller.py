from app.utils.image_loader import load_images
from app.service.embedding import embed_images  # service import
from app.schemas.album_schema import ImageRequest
from fastapi import Request


def embed_controller(req: ImageRequest, request: Request):
    filenames = req.images
    images = load_images(filenames)

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
