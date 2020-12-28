import pytest
import torch

from anatome import utils


def test_normalize_denormalize():
    input = torch.randn(4, 3, 5, 5)
    mean = torch.as_tensor([0.5, 0.5, 0.5])
    std = torch.as_tensor([0.5, 0.5, 0.5])
    output = utils._denormalize(utils._normalize(input, mean, std),
                                mean, std)
    assert torch.allclose(input, output, atol=1e-4)


def test_fft_shift():
    input = torch.randn(4, 3, 8, 8)
    fft = utils.fft_shift(input.rfft(2, normalized=True, onesided=False))
    ifft = utils.ifft_shift(fft).irfft(2, normalized=True, onesided=False)
    assert torch.allclose(input, ifft, atol=1e-4)


@pytest.mark.skipif(not utils.HAS_FFT_MODULE and hasattr(torch, "rfft"), reason="")
@pytest.mark.parametrize("onesided", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("signal_ndim", [2])
def test_rfft(signal_ndim, normalized, onesided):
    input = torch.randn(4, 3, 8, 8)
    assert torch.allclose(input.rfft(signal_ndim, normalized, onesided),
                          utils._rfft(input, signal_ndim, normalized, onesided), atol=1e-3)
