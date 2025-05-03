import torch

_aesthetic_regressor = None


class AestheticRegressor(torch.nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc = torch.nn.Linear(dim, 1)
        self.load_state_dict(torch.load("app/model/aesthetic_regressor.pth"))

    def forward(self, x):
        return self.fc(x).squeeze(1)


def loader_aesthetic_regressor():
    global _aesthetic_regressor
    if _aesthetic_regressor is None:
        _aesthetic_regressor = AestheticRegressor()
        _aesthetic_regressor.eval()
    return _aesthetic_regressor
