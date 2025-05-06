import os
from contextlib import asynccontextmanager

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JOBLIB_NUM_THREADS"] = "1"

import torch
from fastapi import FastAPI

from app.api import api_router
from app.config.settings import IMAGE_MODE
from app.middleware.error_handler import setup_exception_handler
from app.model.aesthetic_regressor import loader_aesthetic_regressor
from app.model.arcface_loader import load_arcface_model
from app.model.clip_loader import load_clip_model
from app.model.yolo_detector_loader import load_yolo_detector
from app.utils.image_loader import get_image_loader

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 실행 시, 모델 및 이미지 로더 초기화 로직입니다."""
    clip_model, clip_preprocess = load_clip_model()
    aesthetic_regressor = loader_aesthetic_regressor()
    arcface_model = load_arcface_model()
    yolo_detector = load_yolo_detector()

    app.state.clip_model = clip_model
    app.state.clip_preprocess = clip_preprocess
    app.state.aesthetic_regressor = aesthetic_regressor
    app.state.arcface_model = arcface_model
    app.state.yolo_detector = yolo_detector
    app.state.image_loader = get_image_loader(IMAGE_MODE)
    yield

app = FastAPI(lifespan=lifespan)
torch.set_num_threads(1)

setup_exception_handler(app)

app.include_router(api_router)
