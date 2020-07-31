import torch
from torch import nn
from torch.nn import functional as F

from anatome import fourier


def test_landscape():
    model = nn.Sequential(nn.Conv2d(3, 4, 3),
                          nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1),
                          nn.Flatten(),
                          nn.Linear(4, 3))
    data = (torch.randn(10, 3, 16, 16),
            torch.randint(2, (10,)))
    fourier.fourier_map(model, data, F.cross_entropy, 4)
