# from functools import partial
# from typing import Any
# import asyncio

# from fastapi import Request

# from app.schemas.album_schema import ImageRequest
# from app.service.people import cluster_faces
# from app.utils.logging_decorator import log_exception, log_flow

# PEOPLE_SEMAPHORE_SIZE = 5
# people_semaphore = asyncio.Semaphore(PEOPLE_SEMAPHORE_SIZE)

# @log_flow
# async def people_controller(req: ImageRequest, request: Request) -> dict[str, Any]:
#     """
#     이미지 파일들의 이름을 받아 동일 인물 기준으로 클러스터링합니다.

#     Args:
#         req (ImageRequest): 이미지 파일 이름 리스트를 포함한 요청 객체입니다.
#         request (Request): FastAPI 요청 객체. 모델 등 앱 상태에 접근하는 데 사용됩니다.

#     Returns:
#         dict: 클러스터링 결과를 포함한 응답. 예시:
#             {
#                 "message": "success",
#                 "data": [
#                     ["img001.jpg", "img005.jpg"],  # 동일 인물 A
#                     ["img002.jpg"],                # 동일 인물 B
#                     ...
#                 ]
#             }

#     """
#     filenames = req.images
#     loop = request.app.state.loop

#     image_loader = request.app.state.image_loader
#     images = await image_loader.load_images(filenames)

#     arcface_model = request.app.state.arcface_model
#     yolo_detector = request.app.state.yolo_detector

#     task_func = partial(cluster_faces, images, filenames, arcface_model, yolo_detector)
    
#     async with people_semaphore:
#         clustering_result = await loop.run_in_executor(None, task_func)

#     return {"message": "success", "data": clustering_result}
