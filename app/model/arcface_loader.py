import torch
from app.model.InsightFace_PyTorch.models import ResNet, IRBlock
from app.model.InsightFace_PyTorch.config import im_size

device = "cpu"  # or "cuda" if available
_arcface_model = None


def load_arcface_model(weight_path: str = "app/model/insight-face-v3.pt"):
    global _arcface_model
    if _arcface_model is None:
        model = ResNet(IRBlock, [3, 4, 23, 3], use_se=True, im_size=im_size)
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval().to(device)
        _arcface_model = model
    return _arcface_model
