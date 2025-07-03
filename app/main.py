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
from aiokafka.errors import KafkaConnectionError

from app.config.secret_loader import load_secrets_from_gcp
# load_secrets_from_gcp()

from app.api import api_router
from app.kafka.consumer import create_kafka_consumer, consume_loop
from app.kafka.producer import create_kafka_producer
from app.config.app_config import get_config
from app.config.settings import IMAGE_MODE, MODEL_NAME, MODEL_BASE_PATH, CATEGORY_FEATURES_FILENAME, QUALITY_FEATURES_FILENAME
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
    config = get_config()
    await config.initialize()

    consumer = create_kafka_consumer(bootstrap_servers=config.kafka_bootstrap_servers)
    producer = create_kafka_producer(bootstrap_servers=config.kafka_bootstrap_servers)

    try:
        await consumer.start()
        print(f"kafka Consumer 연결 성공")
    except KafkaConnectionError as e:
        print(f"kafka Consumer 연결 실패: {e}")
        
    try:
        await producer.start()
        print(f"kafka Producer 연결 성공")
    except KafkaConnectionError as e:
        print(f"kafka Producer 연결 실패: {e}")


    # Kafka 백그라운드 태스크 실행
    kafka_consumer_task = asyncio.create_task(consume_loop(consumer, producer))

    try:
        yield
    finally:
        """서버 종료 시, 리소스 해제 로직입니다."""
        await config.cleanup()
        await consumer.stop()
        await producer.stop()
        kafka_consumer_task.cancel()
        try:
            await kafka_consumer_task
        except asyncio.CancelledError:
            pass
    

app = FastAPI(lifespan=lifespan)
torch.set_num_threads(1)

setup_exception_handler(app)

app.include_router(api_router)
