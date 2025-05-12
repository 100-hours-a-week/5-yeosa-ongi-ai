import torch

_aesthetic_regressor = None


class AestheticRegressor(torch.nn.Module):
    def __init__(self, model_name='ViT-B/32'):
        super().__init__()
        self.fc = torch.nn.Linear(512 if model_name == 'ViT-B/32' else 768, 1)
        self.load_state_dict(torch.load(f"app/model/aesthetic_regressor_{model_name.split('/')[0]}.pth"))

    def forward(self, x):
        return self.fc(x).squeeze(1)


def loader_aesthetic_regressor():
    global _aesthetic_regressor
    if _aesthetic_regressor is None:
        _aesthetic_regressor = AestheticRegressor()
        _aesthetic_regressor.eval()
    return _aesthetic_regressor
