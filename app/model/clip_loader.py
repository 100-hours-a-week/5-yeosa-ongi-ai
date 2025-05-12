import clip

from app.model.clip_preprocess import clip_preprocess_np

_model = None
_preprocess = None


def load_clip_model(device="cpu"):
    global _model, _preprocess
    if _model is None or _preprocess is None:
        _model, _ = clip.load("ViT-B/32", device=device)
        _preprocess = clip_preprocess_np
        _model.eval()
    return _model, _preprocess
