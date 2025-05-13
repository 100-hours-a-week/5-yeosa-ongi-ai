import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from app.config.settings import MODEL_BASE_PATH, AESTHETIC_REGRESSOR_FILENAME

logger = logging.getLogger(__name__)

# 모델 차원 정의
MODEL_DIMENSIONS: Dict[str, int] = {
    'ViT-B/32': 512,
    'ViT-L/14': 768,
}

# 전역 모델 인스턴스
_aesthetic_regressor: Optional['AestheticRegressor'] = None


class AestheticRegressor(nn.Module):
    """
    이미지의 미적 품질을 평가하는 회귀 모델입니다.
    
    이 클래스는 이미지 임베딩을 입력으로 받아 미적 점수를 예측합니다.
    CLIP 모델의 임베딩 차원에 맞춰 선형 레이어를 사용합니다.
    """

    def __init__(self, model_name: str = 'ViT-B/32') -> None:
        """
        AestheticRegressor 인스턴스를 초기화합니다.
        
        Args:
            model_name: 사용할 CLIP 모델의 이름 ('ViT-B/32' 또는 'ViT-L/14')
            
        Raises:
            KeyError: 지원하지 않는 모델 이름이 주어진 경우
            FileNotFoundError: 모델 가중치 파일을 찾을 수 없는 경우

        """
        super().__init__()

        if model_name not in MODEL_DIMENSIONS:
            raise KeyError(f"지원하지 않는 모델 이름: {model_name}")

        model_path = os.path.join(MODEL_BASE_PATH, AESTHETIC_REGRESSOR_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 가중치 파일을 찾을 수 없음: {model_path}")

        self.fc = nn.Linear(MODEL_DIMENSIONS[model_name], 1)
        self.load_state_dict(torch.load(model_path))
        logger.info(
            "AestheticRegressor 초기화 완료",
            extra={"model_name": model_name},
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 임베딩에 대한 미적 점수를 예측합니다.
        
        Args:
            x: 이미지 임베딩 텐서 (batch_size x embedding_dim)
            
        Returns:
            torch.Tensor: 예측된 미적 점수 (batch_size)
        """
        return self.fc(x).squeeze(1)


def load_aesthetic_regressor(model_name: str = 'ViT-B/32') -> AestheticRegressor:
    """
    AestheticRegressor 인스턴스를 로드하거나 생성합니다.
    
    싱글톤 패턴을 사용하여 모델 인스턴스를 재사용합니다.
    
    Args:
        model_name: 사용할 CLIP 모델의 이름 ('ViT-B/32' 또는 'ViT-L/14')
        
    Returns:
        AestheticRegressor: 초기화된 모델 인스턴스
        
    Raises:
        KeyError: 지원하지 않는 모델 이름이 주어진 경우
        FileNotFoundError: 모델 가중치 파일을 찾을 수 없는 경우

    """
    global _aesthetic_regressor
    
    if _aesthetic_regressor is None:
        logger.info(
            "AestheticRegressor 인스턴스 생성",
            extra={"model_name": model_name},
        )
        _aesthetic_regressor = AestheticRegressor(model_name)
        _aesthetic_regressor.eval()
    else:
        logger.debug("기존 AestheticRegressor 인스턴스 재사용")
        
    return _aesthetic_regressor
