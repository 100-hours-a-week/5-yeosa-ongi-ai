import cv2
import numpy as np
import torch


def get_clip_preprocess_fn(device: str) -> callable:
    """
    CLIP 전처리 함수 반환.
    GPU 사용 시 GPU 텐서로 변환.
    """
    def clip_preprocess_np(img: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(3, 1, 1)
        return (tensor - mean) / std
    return clip_preprocess_np
