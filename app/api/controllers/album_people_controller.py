from app.utils.image_loader import get_image_loader
from app.schemas.album_schema import ImageRequest
from app.service.people import cluster_faces
from app.utils.logging_decorator import log_exception
from fastapi import Request

@log_exception
async def people_controller(req: ImageRequest, request: Request):
    filenames = req.images
    
    image_loader = request.app.state.image_loader
    images = await image_loader.load_images(filenames)

    arcface_model = request.app.state.arcface_model
    yolo_detector = request.app.state.yolo_detector

    clustering_result = cluster_faces(
        images, filenames, arcface_model, yolo_detector
    )

    return {"message": "success", "data": clustering_result}
