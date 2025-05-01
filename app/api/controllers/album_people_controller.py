from app.utils.image_loader import get_image_loader
from app.schemas.album_schema import ImageRequest
from app.service.people import cluster_faces
from fastapi import Request


def people_controller(req: ImageRequest, request: Request):
    filenames = req.images
    
    image_loader = get_image_loader()
    images = image_loader.load_images(filenames)

    arcface_model = request.app.state.arcface_model
    yolo_detector = request.app.state.yolo_detector

    clustering_result = cluster_faces(
        images, filenames, arcface_model, yolo_detector
    )

    return {"message": "success", "data": clustering_result}
