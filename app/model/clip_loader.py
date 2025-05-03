import clip

_model = None
_preprocess = None


def load_clip_model(device="cpu"):
    global _model, _preprocess
    if _model is None or _preprocess is None:
        _model, _preprocess = clip.load("ViT-B/32", device=device)
        _model.eval()
    return _model, _preprocess
