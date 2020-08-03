import torch

from anatome import utils


def test_normalize_denormalize():
    input = torch.randn(1, 3, 5, 5)
    mean = torch.as_tensor([0.5, 0.5, 0.5])
    std = torch.as_tensor([0.5, 0.5, 0.5])
    output = utils._denormalize(utils._normalize(input, mean, std),
                                mean, std)
    assert torch.allclose(input, output)


def test_fft_shift():
    input = torch.randn(1, 3, 8, 8)
    fft = utils.fft_shift(input.rfft(2, normalized=True, onesided=False))
    ifft = utils.ifft_shift(fft).irfft(2, normalized=True, onesided=False)
    assert torch.allclose(input, ifft, atol=1e-6)
