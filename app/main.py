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

MAX_WORKERS = 8

GPU_SERVER_BASE_URL = os.getenv("GPU_SERVER_BASE_URL")
if not GPU_SERVER_BASE_URL:
    raise EnvironmentError("GPU_SERVER_BASE_URL이 .env 파일에 없습니다.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 실행 시, 모델 및 이미지 로더 초기화 로직입니다."""
    config = get_config()
    await config.initialize()

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
