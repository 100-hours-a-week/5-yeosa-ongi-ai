import torch

_aesthetic_regressor = None

dim = {'ViT-B/32': 512, 'ViT-L/14': 768}

class AestheticRegressor(torch.nn.Module):
    def __init__(self, model_name='ViT-B/32'):
        super().__init__()
        self.fc = torch.nn.Linear(dim[model_name], 1)
        self.load_state_dict(torch.load(f"app/model/{model_name}/aesthetic_regressor.pth"))

    def forward(self, x):
        return self.fc(x).squeeze(1)


def loader_aesthetic_regressor():
    global _aesthetic_regressor
    if _aesthetic_regressor is None:
        _aesthetic_regressor = AestheticRegressor()
        _aesthetic_regressor.eval()
    return _aesthetic_regressor
