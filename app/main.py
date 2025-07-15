import os
import asyncio
import httpx
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JOBLIB_NUM_THREADS"] = "1"

import torch
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.config.secret_loader import load_secrets_from_gcp
load_secrets_from_gcp()

from app.api import api_router
from app.config.app_config import get_config
from app.middleware.error_handler import setup_exception_handler
<<<<<<< HEAD
=======
from app.model.aesthetic_regressor import load_aesthetic_regressor
from app.utils.image_loader import (
    get_image_loader,
    GCSImageLoader,
    S3ImageLoader,
)
import logging

logger = logging.getLogger(__name__)
>>>>>>> main

MAX_WORKERS = 8

GPU_SERVER_BASE_URL = os.getenv("GPU_SERVER_BASE_URL")
if not GPU_SERVER_BASE_URL:
    raise EnvironmentError("GPU_SERVER_BASE_URL이 .env 파일에 없습니다.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 실행 시, 모델 및 이미지 로더 초기화 로직입니다."""
<<<<<<< HEAD
    config = get_config()
    await config.initialize()
=======
    aesthetic_regressor = load_aesthetic_regressor(MODEL_NAME)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    loop.set_default_executor(executor)
    category_data = torch.load(os.path.join(MODEL_BASE_PATH, CATEGORY_FEATURES_FILENAME), weights_only=False)
    parent_categories = category_data["parent_categories"]
    parent_embeds = category_data["parent_embeds"]
    embed_dict = category_data["embed_dict"]
    category_dict = category_data["category_dict"]
    quality_data = torch.load(os.path.join(MODEL_BASE_PATH, QUALITY_FEATURES_FILENAME), weights_only=True)
    quality_text_features = quality_data["text_features"]
    quality_fields = quality_data["fields"]

    app.state.aesthetic_regressor = aesthetic_regressor
    app.state.image_loader = get_image_loader(IMAGE_MODE)
    app.state.loop = loop
    app.state.parent_categories = parent_categories
    app.state.parent_embeds = parent_embeds
    app.state.embed_dict = embed_dict
    app.state.category_dict = category_dict
    app.state.quality_text_features = quality_text_features
    app.state.quality_fields = quality_fields
    app.state.redis = init_redis()
    logger.info(f"GPU_SERVER_BASE_URL : {GPU_SERVER_BASE_URL}")
    app.state.gpu_client = httpx.AsyncClient(
        base_url=GPU_SERVER_BASE_URL,
        timeout=60.0,
        headers={"Content-Type": "application/json"},
    )
>>>>>>> main

    try:
        yield
    except Exception as e:
        print(f"[lifespan] 예외 발생: {e}", flush=True)
        raise
    finally:
        try:
            await config.cleanup()
        except Exception as e:
            print(f"[cleanup] 예외 발생: {e}", flush=True)

    

app = FastAPI(lifespan=lifespan)
torch.set_num_threads(1)

setup_exception_handler(app)

app.include_router(api_router)

Instrumentator().instrument(app).expose(app)
