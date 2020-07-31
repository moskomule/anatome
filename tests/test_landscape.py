import torch
from torch import nn
from torch.nn import functional as F

from anatome import landscape


def test_landscape():
    model = nn.Sequential(nn.Linear(4, 3),
                          nn.ReLU(),
                          nn.Linear(3, 2))
    data = (torch.randn(10, 4),
            torch.randint(2, (10,)))
    x_coord, y = landscape.landscape1d(model, data, F.cross_entropy, (-1, 1), 0.5)
    assert x_coord.shape == y.shape

    x_coord, y_coord, z = landscape.landscape2d(model, data, F.cross_entropy, (-1, 1), (-1, 1), (0.5, 0.5))
    assert x_coord.shape == y_coord.shape
    assert x_coord.shape == z.shape
