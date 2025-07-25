import os
import asyncio, requests, tempfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import aiofiles
import aioboto3
import cv2
import numpy as np
from dotenv import load_dotenv
from botocore.config import Config
from gcloud.aio.storage import Storage

from app.config.settings import ImageMode

load_dotenv()

# local
LOCAL_IMG_PATH_raw = os.getenv("LOCAL_IMG_PATH")

# GCS
GCS_BUCKET_NAME_raw = os.getenv("GCS_BUCKET_NAME")
GCP_KEY_raw = os.getenv("GCP_KEY")

# S3
S3_BUCKET_NAME_raw = os.getenv("S3_BUCKET_NAME")
AWS_ACCESS_KEY_ID_raw = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY_raw = os.getenv("AWS_SECRET_KEY")
AWS_REGION_raw = os.getenv("AWS_REGION")


# local
if LOCAL_IMG_PATH_raw is None:
    raise EnvironmentError("LOCAL_IMG_PATH은 .env에 설정되어야 합니다.")
# # GCS
# if GCS_BUCKET_NAME_raw is None:
#     raise EnvironmentError("BUCKET_NAME은 .env에 설정되어야 합니다.")
# if GCP_KEY_raw is None:
#     raise EnvironmentError("GCP_KEY_PATH은 .env에 설정되어야 합니다.")

# S3
if S3_BUCKET_NAME_raw is None:
    raise EnvironmentError("S3_BUCKET_NAME은 .env에 설정되어야 합니다.")
if AWS_ACCESS_KEY_ID_raw is None or AWS_SECRET_ACCESS_KEY_raw is None:
    raise EnvironmentError("AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY는 .env에 설정되어야 합니다.")
if AWS_REGION_raw is None:
    raise EnvironmentError("AWS_REGION은 .env에 설정되어야 합니다.")

# 타입이 str로 확정됨 (mypy 추론 가능)
LOCAL_IMG_PATH: str = LOCAL_IMG_PATH_raw

# GCS
if GCS_BUCKET_NAME_raw:
    GCS_BUCKET_NAME: str = GCS_BUCKET_NAME_raw
if GCP_KEY_raw:
    GCP_KEY: str = GCP_KEY_raw

S3_BUCKET_NAME: str = S3_BUCKET_NAME_raw
AWS_ACCESS_KEY_ID: str = AWS_ACCESS_KEY_ID_raw
AWS_SECRET_ACCESS_KEY: str = AWS_SECRET_ACCESS_KEY_raw
AWS_REGION: str = AWS_REGION_raw

# 공통 디코더
def decode_image_cv2(image_bytes: bytes, label: str, scale: str = 'RGB') -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    flags = cv2.IMREAD_GRAYSCALE if scale == 'GRAY' else cv2.IMREAD_COLOR
    img = cv2.imdecode(nparr, flags)
    if scale == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

class BaseImageLoader(ABC):
    """
    이미지 로더의 추상 베이스 클래스.

    모든 이미지 로더는 `load_images` 메서드를 구현해야 합니다.
    """

    @abstractmethod
    async def load_images(self, filenames: list[str], scale: list[str]) -> list[bytes]:
        """
        주어진 이미지 파일 이름 리스트에 대해 이미지를 로드합니다.

        Args:
            filenames (list[str]): 로드할 이미지 파일 이름 목록
            scale: RGB / GRAY

        Returns:
            list[bytes]: 로드된 이미지 바이트 리스트

        """
        pass


class LocalImageLoader(BaseImageLoader):
    """로컬 파일 시스템에서 이미지를 로드하는 클래스입니다."""

    def __init__(self, image_dir: str = LOCAL_IMG_PATH):
        """
        Args:
            image_dir (str): 이미지가 저장된 로컬 디렉토리 경로

        """
        self.image_dir = image_dir

    async def _load_single_image(self, filename: str, scale: list[str] = 'RGB') -> np.ndarray:
        file_path = os.path.join(self.image_dir, filename)

        # 1. 파일 비동기 I/O로 읽기
        async with aiofiles.open(file_path, mode="rb") as f:
            image_bytes = await f.read()

        # 2. 디코딩은 스레드에서 실행
        loop = asyncio.get_running_loop()
        decoded_img = await loop.run_in_executor(None, decode_image_cv2, image_bytes, "local", scale)
        return decoded_img

    async def load_images(self, filenames: list[str], scale: list[str] = 'RGB') -> list[np.ndarray]:
        """
        비동기적으로 로컬 이미지들을 로드합니다.

        Args:
            filenames (list[str]): 이미지 파일 이름 리스트

        Returns:
            list[np.ndarray]: 로드된 이미지 리스트

        """
        tasks = [self._load_single_image(name, scale) for name in filenames]
        return await asyncio.gather(*tasks)


class GCSImageLoader(BaseImageLoader):
    """Google Cloud Storage(GCS)에서 이미지를 로드하는 클래스입니다."""

    def __init__(
        self, bucket_name: str = GCS_BUCKET_NAME, gcp_key: str = GCP_KEY
    ):
        """
        Args:
            bucket_name (str): GCS 버킷 이름
            key_path (str): 서비스 계정 키 경로 (.json)

        """
        if GCP_KEY_raw:
            self.client = Storage(service_file=GCP_KEY_raw)
        else:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
                tmp.write(gcp_key)
                self._temp_key_path = tmp.name
            self.client = Storage(service_file=self._temp_key_path)
        self.bucket_name = bucket_name

    async def _download(self, file_name: str) -> bytes:
        """
        GCS에서 단일 이미지를 다운로드하고 RGB로 변환합니다.

        Args:
            file_name (str): GCS 내 파일 이름

        Returns:
            bytes: 로드된 이미지 바이트

        """
        image_bytes = await self.client.download(
            bucket=self.bucket_name, object_name=file_name
        )
        return image_bytes

    async def _process_single_file(
        self, filename: str, executor=None, scale: list[str]='RGB'
    ) -> np.ndarray:
        loop = asyncio.get_running_loop()
        image_bytes = await self._download(filename)
        decoded_img = await loop.run_in_executor(
            executor, decode_image_cv2, image_bytes, "gcs", scale
        )
        return decoded_img

    async def load_images(self, filenames: list[str], scale: list[str] = 'RGB') -> list[np.ndarray]:
        """
        비동기적으로 GCS에서 이미지를 병렬로 다운로드합니다.

        Args:
            filenames (list[str]): GCS 상의 이미지 파일 이름 리스트

        Returns:
            list[np.ndarray]: 로드된 이미지 리스트

        """
        #start = time.time()
        tasks = [self._process_single_file(f, None, scale) for f in filenames]

        result = await asyncio.gather(*tasks)
        return result

class S3ImageLoader(BaseImageLoader):
    """Amazon S3에서 이미지를 로드하는 클래스입니다."""

    def __init__(
        self,
        bucket_name: str = S3_BUCKET_NAME,
        aws_access_key_id: str = AWS_ACCESS_KEY_ID,
        aws_secret_access_key: str = AWS_SECRET_ACCESS_KEY,
        region_name: str = AWS_REGION,
    ):
        """
        Args:
            bucket_name (str): S3 버킷 이름
            aws_access_key_id (str): AWS 액세스 키 ID
            aws_secret_access_key (str): AWS 시크릿 액세스 키
            region_name (str): S3 버킷의 리전 이름

        """
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

    async def _download(self, file_ref: str) -> bytes:
        """
        S3에서 단일 이미지를 다운로드하고 RGB로 변환합니다.

        Args:
            file_ref (str): S3 내 파일 이름 (key)

        Returns:
            bytes: 로드된 이미지 바이트

        """
        response = await self.client.get_object(
            Bucket=self.bucket_name, Key=file_ref
        )
        image_bytes = await response["Body"].read()

        return image_bytes
    
    async def _process_single_file(self, filename: str, scale: list[str] = 'RGB') -> np.ndarray:
        loop = asyncio.get_running_loop()
        image_bytes = await self._download(filename)
        decoded = await loop.run_in_executor(
            None, decode_image_cv2, image_bytes, "s3", scale
        )
        return decoded

    async def load_images(self, filenames: list[str], scale: list[str] = 'RGB') -> list[np.ndarray]:
        """
        비동기적으로 S3에서 이미지를 병렬로 다운로드합니다.

        Args:
            filenames (list[str]): S3 내 이미지 파일 이름 리스트

        Returns:
            list[np.ndarray]: 로드된 이미지 바이트 리스트

        """
        tasks = [self._process_single_file(f, scale) for f in filenames]
        result = await asyncio.gather(*tasks)
        return result


def get_image_loader(mode: ImageMode) -> BaseImageLoader:
    """
    이미지 로딩 모드를 기반으로 적절한 이미지 로더 인스턴스를 반환합니다.

    Args:
        mode (ImageMode): 이미지 로딩 방식 (로컬 or GCS or S3)

    Returns:
        BaseImageLoader: 선택된 이미지 로더 인스턴스

    """
    if mode == ImageMode.GCS:
        return GCSImageLoader()
    elif mode == ImageMode.S3:
        return S3ImageLoader()
    return LocalImageLoader()
