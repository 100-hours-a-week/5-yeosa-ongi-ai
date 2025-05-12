import logging
from typing import List, Any, Callable

import torch
from PIL import Image

from app.core.cache import set_cached_embedding
from app.utils.logging_decorator import log_exception, log_flow

logger = logging.getLogger(__name__)


@log_flow
def embed_images(
    model: torch.nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    images: List[Image.Image],
    filenames: List[str],
    batch_size: int = 16,
    device: str = "cpu",
) -> None:
    """
    이미지 배치를 임베딩하고 캐시에 저장합니다.

    Args:
        model: 이미지 인코딩 모델
        preprocess: 이미지 전처리 함수
        images: 임베딩할 이미지 리스트
        filenames: 이미지 파일명 리스트
        batch_size: 배치 크기 (기본값: 16)
        device: 연산에 사용할 디바이스 (기본값: "cpu")

    Returns:
        None

    """
    logger.info(
        "이미지 임베딩 시작",
        extra={
            "total_images": len(images),
            "batch_size": batch_size,
            "device": device,
        },
    )

    preprocessed_img = [preprocess(image) for image in images]

    for i in range(0, len(images), batch_size):
        batch_images = preprocessed_img[i : i + batch_size]
        batch_filenames = filenames[i : i + batch_size]

        logger.debug(
            "배치 처리 중",
            extra={
                "batch_start": i,
                "batch_end": i + batch_size,
                "batch_size": len(batch_images),
            },
        )

        image_input = torch.stack(batch_images).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(image_input).cpu()

        for filename, feature in zip(batch_filenames, batch_features):
            set_cached_embedding(filename, feature)

    logger.info("이미지 임베딩 완료", extra={"total_images": len(images)})
    return None
