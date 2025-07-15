# app/config/app_config.py

import os
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from loguru import logger

import torch

from app.config.redis import init_redis
from app.config.settings import (
    IMAGE_MODE, MODEL_NAME, MODEL_BASE_PATH,
    CATEGORY_FEATURES_FILENAME, QUALITY_FEATURES_FILENAME,
)
from app.config.kafka_config import KAFKA_GROUP_ID_MAP
from app.model.aesthetic_regressor import load_aesthetic_regressor
from app.utils.image_loader import get_image_loader, S3ImageLoader, GCSImageLoader


class AppConfig:
    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.aesthetic_regressor = None
        self.image_loader = None
        self.parent_categories = None
        self.parent_embeds = None
        self.embed_dict = None
        self.category_dict = None
        self.quality_text_features = None
        self.quality_fields = None
        self.redis = None
        self.redis_semaphore = None
        self.gpu_client: Optional[httpx.AsyncClient] = None
        self.kafka_bootstrap_servers: Optional[str] = None
        self.kafka_tasks: list[asyncio.Task] = []

    async def initialize(self):
        from app.kafka.consumer import run_kafka_consumer, ALL_TOPICS

        self.loop = asyncio.get_running_loop()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.loop.set_default_executor(self.executor)

        # 동기 작업 비동기 실행
        self.aesthetic_regressor = await self.loop.run_in_executor(
            None, load_aesthetic_regressor, MODEL_NAME
        )

        category_data = await self.loop.run_in_executor(
            None,
            lambda: torch.load(
                os.path.join(MODEL_BASE_PATH, CATEGORY_FEATURES_FILENAME),
                weights_only=False
            )
        )

        self.parent_categories = category_data["parent_categories"]
        self.parent_embeds = category_data["parent_embeds"]
        self.embed_dict = category_data["embed_dict"]
        self.category_dict = category_data["category_dict"]

        quality_data = await self.loop.run_in_executor(
            None, torch.load,
            os.path.join(MODEL_BASE_PATH, QUALITY_FEATURES_FILENAME)
        )
        self.quality_text_features = quality_data["text_features"]
        self.quality_fields = quality_data["fields"]

        self.image_loader = get_image_loader(IMAGE_MODE)
        if IMAGE_MODE == IMAGE_MODE.S3 and isinstance(self.image_loader, S3ImageLoader):
            await self.image_loader.init_client()

        self.redis = init_redis()
        self.redis_semaphore = asyncio.Semaphore(80)
        
        try:
            if await self.redis.ping():
                logger.info("Redis 연결 성공")
        except Exception as e:
            logger.error(f"Redis 연결 실패: {e}")

        gpu_server_base_url = os.getenv("GPU_SERVER_BASE_URL")
        if not gpu_server_base_url:
            raise EnvironmentError("GPU_SERVER_BASE_URL이 .env 파일에 없습니다.")

        self.gpu_client = httpx.AsyncClient(
            base_url=gpu_server_base_url,
            timeout=60.0,
            headers={"Content-Type": "application/json"},
        )

        # Kafka 컨슈머 루프 등록 (두 그룹 모두 실행)
        for topic in ALL_TOPICS:
            group_id = KAFKA_GROUP_ID_MAP[topic]
            task = asyncio.create_task(run_kafka_consumer(topic, group_id))
            self.kafka_tasks.append(task)

    async def cleanup(self):
        for task in self.kafka_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                print("Kafka consumer task cancelled cleanly.")

        if IMAGE_MODE == IMAGE_MODE.GCS and isinstance(self.image_loader, GCSImageLoader):
            await self.image_loader.client.close()
            temp_path = getattr(self.image_loader, "_temp_file_path", None)
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"임시 GCP 키 파일 삭제됨: {temp_path}")

        if IMAGE_MODE == IMAGE_MODE.S3 and isinstance(self.image_loader, S3ImageLoader):
            await self.image_loader.close_client()

    def get_executor(self):
        return self.executor

    def get_loop(self):
        return self.loop


app_config = AppConfig()

def get_config() -> AppConfig:
    return app_config
