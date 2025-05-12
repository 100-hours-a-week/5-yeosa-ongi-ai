import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import aiofiles

import aioboto3
import numpy as np
import cv2
from dotenv import load_dotenv
from turbojpeg import TurboJPEG, TJCS_RGB
from gcloud.aio.storage import Storage
import asyncio
from functools import partial
from botocore.config import Config
from app.config.settings import ImageMode

load_dotenv()

# 환경 변수 로드
LOCAL_IMG_PATH = os.getenv("LOCAL_IMG_PATH")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCP_KEY_PATH = os.getenv("GCP_KEY_PATH")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")

# 환경 변수 검증
required_envs = {
    "LOCAL_IMG_PATH": LOCAL_IMG_PATH,
    "GCS_BUCKET_NAME": GCS_BUCKET_NAME,
    "GCP_KEY_PATH": GCP_KEY_PATH,
    "S3_BUCKET_NAME": S3_BUCKET_NAME,
    "AWS_ACCESS_KEY": AWS_ACCESS_KEY_ID,
    "AWS_SECRET_KEY": AWS_SECRET_ACCESS_KEY,
    "AWS_REGION": AWS_REGION,
}

for name, value in required_envs.items():
    if value is None:
        raise EnvironmentError(f"{name}은(는) .env에 설정되어야 합니다.")

GCS_DOWNLOAD_SEMAPHORE_SIZE = 10
gcs_download_semaphore = asyncio.Semaphore(GCS_DOWNLOAD_SEMAPHORE_SIZE)

# 전역 스레드 풀
executor = ThreadPoolExecutor(max_workers=10)


class BaseImageLoader(ABC):
    @abstractmethod
    async def load_images(self, filenames: list[str]) -> list[np.ndarray]:
        pass


jpeg = TurboJPEG()


def decode_image_cv2(image_bytes: bytes, label: str) -> np.ndarray:
    start = time.time()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    end = time.time()
    start2 = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    end2 = time.time()
    print(f"디코딩 시간({label}): {end - start:.6f}초")
    print(f"색상공간 변환 시간({label}): {end2 - start2:.6f}초")
    return img


def decode_image_turbojpeg(image_bytes: bytes, label: str) -> np.ndarray:
    start = time.time()
    img = jpeg.decode(image_bytes, pixel_format=TJCS_RGB)  # 자동 RGB
    end = time.time()
    print(f"turbojpeg 디코딩 시간({label}): {end - start:.6f}초")
    return img


class GCSImageLoader(BaseImageLoader):
    def __init__(
        self, bucket_name: str = GCS_BUCKET_NAME, key_path: str = GCP_KEY_PATH
    ):
        self.client = Storage(service_file=key_path)
        self.bucket_name = bucket_name

    async def _download(self, file_name: str) -> bytes:
        async with gcs_download_semaphore:
            start = time.time()
            image_bytes = await self.client.download(
                bucket=self.bucket_name, object_name=file_name
            )
            # image_bytes = await blob.download_as_bytes(client=self.client)
            end = time.time()
            print(f" GCS 다운로드 시간 : {end - start}")
            return image_bytes

    async def _process_single_file(
        self, filename: str, executor=None
    ) -> np.ndarray:
        loop = asyncio.get_running_loop()
        image_bytes = await self._download(filename)
        decoded_img = await loop.run_in_executor(
            executor, decode_image_turbojpeg, image_bytes, "gcs"
        )
        return decoded_img

    async def load_images(self, filenames: list[str]) -> list[np.ndarray]:
        start = time.time()
        tasks = [self._process_single_file(f, executor) for f in filenames]

        result = await asyncio.gather(*tasks)
        end = time.time()
        print(f"전체 시간(gcs): {end - start:.6f}초")
        return result


class S3ImageLoader(BaseImageLoader):
    def __init__(
        self,
        bucket_name: str = S3_BUCKET_NAME,
        aws_access_key_id: str = AWS_ACCESS_KEY_ID,
        aws_secret_access_key: str = AWS_SECRET_ACCESS_KEY,
        region_name: str = AWS_REGION,
    ):
        self.bucket_name = bucket_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.session = aioboto3.Session()
        self.client = None

    async def init_client(self):
        """S3 클라이언트 초기화 (lifespan 시작 시 호출됨)"""
        self.client = await self.session.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
            config=Config(retries={"max_attempts": 3}),
        ).__aenter__()

    async def close_client(self):
        """S3 클라이언트 종료 (lifespan 끝날 때 호출됨)"""
        if self.client:
            await self.client.__aexit__(None, None, None)
            self.client = None

    async def _download(self, file_name: str) -> bytes:
        response = await self.client.get_object(
            Bucket=self.bucket_name, Key=file_name
        )
        return await response["Body"].read()

    async def _process_single_file(self, filename: str) -> np.ndarray:
        loop = asyncio.get_running_loop()
        image_bytes = await self._download(filename)
        decoded = await loop.run_in_executor(
            executor, decode_image_turbojpeg, image_bytes, "s3"
        )
        return decoded

    async def load_images(self, filenames: list[str]) -> list[np.ndarray]:
        start = time.time()
        tasks = [self._process_single_file(f) for f in filenames]
        result = await asyncio.gather(*tasks)
        end = time.time()
        print(f"전체 시간(s3): {end - start:.6f}초")
        return result


class LocalImageLoader(BaseImageLoader):
    def __init__(self, image_dir: str = LOCAL_IMG_PATH):
        self.image_dir = image_dir

    async def _read_file_async(self, filepath: str) -> bytes:
        async with aiofiles.open(filepath, "rb") as f:
            return await f.read()

    async def _process_single_file(self, filename: str) -> np.ndarray:
        filepath = os.path.join(self.image_dir, filename)
        image_bytes = await self._read_file_async(filepath)
        loop = asyncio.get_running_loop()
        decoded = await loop.run_in_executor(
            executor, decode_image_turbojpeg, image_bytes, "local"
        )
        return decoded

    async def load_images(self, filenames: list[str]) -> list[np.ndarray]:
        start = time.time()
        tasks = [self._process_single_file(f) for f in filenames]

        result = await asyncio.gather(*tasks)
        end = time.time()
        print(f"전체 시간(gcs): {end - start:.6f}초")
        return result


def get_image_loader(mode: ImageMode) -> BaseImageLoader:
    if mode == ImageMode.GCS:
        return GCSImageLoader()
    elif mode == ImageMode.S3:
        return S3ImageLoader()
    elif mode == ImageMode.LOCAL:
        return LocalImageLoader()
    raise NotImplementedError("지원하지 않는 이미지 로딩 모드입니다.")
