import torch
import clip

_model = None
_preprocess = None


def load_clip_model(model_name="ViT-B/32", device=None):
    global _model, _preprocess
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _model is None or _preprocess is None:
        _model, _preprocess = clip.load(model_name, device=device)
        _model.eval()
    return _model, _preprocess
