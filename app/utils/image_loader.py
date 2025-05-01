import os
from PIL import Image
from abc import ABC, abstractmethod
from io import BytesIO
from google.cloud import storage
from starlette.concurrency import run_in_threadpool
from concurrent.futures import ThreadPoolExecutor
from app.config.settings import ImageMode
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

LOCAL_IMG_PATH: Optional[str] = os.getenv("LOCAL_IMG_PATH")
BUCKET_NAME: Optional[str] = os.getenv("BUCKET_NAME")
GCP_KEY_PATH: Optional[str] = os.getenv("GCP_KEY_PATH")

assert LOCAL_IMG_PATH is not None, "LOCAL_IMG_PATH은 .env에 설정되어야 합니다."
assert BUCKET_NAME is not None, "BUCKET_NAME은 .env에 설정되어야 합니다."
assert GCP_KEY_PATH is not None, "KEY_PATH은 .env에 설정되어야 합니다."

class BaseImageLoader(ABC):
    @abstractmethod
    async def load_images(self) -> list[Image.Image]:
        pass

class LocalImageLoader(BaseImageLoader):
    def __init__(self, image_dir: str = LOCAL_IMG_PATH):
        self.image_dir = image_dir 

    async def load_images(self, filenames: list[str]) -> list[Image.Image]:
        return await run_in_threadpool(lambda: list(Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
            for filename in filenames))


class GCSImageLoader(BaseImageLoader):
    def __init__(self, bucket_name: str = BUCKET_NAME, key_path: str = GCP_KEY_PATH):
        self.client = storage.Client.from_service_account_json(key_path)
        self.bucket = self.client.bucket(bucket_name)
        self.executor = ThreadPoolExecutor(max_workers=10)

    def _download(self, file_name):
        blob = self.bucket.blob(file_name)
        image_bytes = blob.download_as_bytes()
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    async def load_images(self, filenames: list[str]) -> list[Image.Image]:
        return await run_in_threadpool(lambda: list(self.executor.map(self._download, filenames)))


def get_image_loader(mode: ImageMode) -> BaseImageLoader:
    if mode == ImageMode.GCS:
        return GCSImageLoader() 
    return LocalImageLoader()