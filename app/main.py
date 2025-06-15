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

from app.api import api_router
from app.config.redis import init_redis
from app.config.secret_loader import load_secrets_from_gcp
from app.config.settings import IMAGE_MODE, MODEL_NAME, MODEL_BASE_PATH, CATEGORY_FEATURES_FILENAME, QUALITY_FEATURES_FILENAME, APP_ENV
from app.middleware.error_handler import setup_exception_handler
from app.model.aesthetic_regressor import load_aesthetic_regressor
from app.utils.image_loader import (
    get_image_loader,
    GCSImageLoader,
    S3ImageLoader,
)

MAX_WORKERS = 8

GPU_SERVER_BASE_URL = os.getenv("GPU_SERVER_BASE_URL")
if not GPU_SERVER_BASE_URL:
    raise EnvironmentError("GPU_SERVER_BASE_URL이 .env 파일에 없습니다.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 실행 시, 모델 및 이미지 로더 초기화 로직입니다."""
    load_secrets_from_gcp()
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
    app.state.gpu_client = httpx.AsyncClient(
        base_url=GPU_SERVER_BASE_URL, 
        timeout=60.0,
        headers={"Content-Type": "application/json"},
    )

    try:
        pong = await app.state.redis.ping()
        if pong:
            print("Redis 연결 성공")
    except Exception as e:
        print(f"Redis 연결 실패: {e}")

    if IMAGE_MODE == IMAGE_MODE.S3:
        if isinstance(app.state.image_loader, S3ImageLoader):
            await app.state.image_loader.init_client()

    yield

    # 서버 종료 시 리소스 해제
    if IMAGE_MODE == IMAGE_MODE.GCS:
        if isinstance(app.state.image_loader, GCSImageLoader):
            await app.state.image_loader.client.close()

            temp_path = getattr(app.state.image_loader, "_temp_file_path", None)
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"임시 GCP 키 파일 삭제됨: {temp_path}")

    if IMAGE_MODE == IMAGE_MODE.S3:
        if isinstance(app.state.image_loader, S3ImageLoader):
            await app.state.image_loader.close_client()

app = FastAPI(lifespan=lifespan)
torch.set_num_threads(1)

setup_exception_handler(app)

app.include_router(api_router)
