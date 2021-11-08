import torch

from anatome import utils


def test_normalize_denormalize():
    input = torch.randn(4, 3, 5, 5)
    mean = torch.as_tensor([0.5, 0.5, 0.5])
    std = torch.as_tensor([0.5, 0.5, 0.5])
    output = utils._denormalize(utils._normalize(input, mean, std),
                                mean, std)
    assert torch.allclose(input, output, atol=1e-4)
