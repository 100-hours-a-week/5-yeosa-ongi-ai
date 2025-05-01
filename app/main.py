import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JOBLIB_NUM_THREADS"] = "1"

from fastapi import FastAPI
import torch
from app.model.clip_loader import load_clip_model
from app.model.aesthetic_regressor import loader_aesthetic_regressor
from app.model.arcface_loader import load_arcface_model
from app.model.yolo_detector_loader import load_yolo_detector
from app.utils.image_loader import GCSImageLoader, LocalImageLoader
from app.config.settings import IMAGE_MODE
from app.api import api_router

app = FastAPI()
torch.set_num_threads(1)


@app.on_event("startup")
def load():
    clip_model, clip_preprocess = load_clip_model()
    aesthetic_regressor = loader_aesthetic_regressor()
    arcface_model = load_arcface_model()
    yolo_detector = load_yolo_detector()

    app.state.clip_model = clip_model
    app.state.clip_preprocess = clip_preprocess
    app.state.aesthetic_regressor = aesthetic_regressor
    app.state.arcface_model = arcface_model
    app.state.yolo_detector = yolo_detector

    if IMAGE_MODE == "gcs":
        app.state.image_loader = GCSImageLoader()
    else:
        app.state.image_loader = LocalImageLoader()


app.include_router(api_router)
