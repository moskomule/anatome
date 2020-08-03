from typing import Tuple, Callable, Optional

import torch
from torch import nn, Tensor

AUTO_CAST = False


def use_auto_cast() -> None:
    """ Enable AMP autocast.
    """
    global AUTO_CAST
    AUTO_CAST = True


def _evaluate(model: nn.Module,
              data: Tuple[Tensor, Tensor],
              criterion: Callable[[Tensor, Tensor], Tensor]
              ) -> float:
    # evaluate model with given data points using the criterion
    with torch.cuda.amp.autocast(AUTO_CAST):
        input, target = data
        return criterion(model(input), target).item()


def _normalize(input: Tensor,
               mean: Tensor,
               std: Tensor
               ) -> Tensor:
    # normalize tensor in [0, 1] to [-1, 1]
    input = input.clone()
    input.add_(-mean[:, None, None]).div_(std[:, None, None])
    return input


def _denormalize(input: Tensor,
                 mean: Tensor,
                 std: Tensor
                 ) -> Tensor:
    # denormalize tensor in [-1, 1] to [0, 1]
    input = input.clone()
    input.mul_(std[:, None, None]).add_(mean[:, None, None])
    return input


def fft_shift(input: torch.Tensor,
              dims: Optional[Tuple[int, ...]] = None
              ) -> torch.Tensor:
    """ PyTorch version of np.fftshift

    Args:
        input: rFFTed Tensor of size [Bx]CxHxWx2
        dims:

    Returns: shifted tensor

    """

    if dims is None:
        dims = [i for i in range(1 if input.dim() == 4 else 2, input.dim() - 1)]  # H, W
    shift = [input.size(dim) // 2 for dim in dims]
    return torch.roll(input, shift, dims)


def ifft_shift(input: torch.Tensor,
               dims: Optional[Tuple[int, ...]] = None
               ) -> torch.Tensor:
    """ PyTorch version of np.ifftshift

    Args:
        input: rFFTed Tensor of size [Bx]CxHxWx2
        dims:

    Returns: shifted tensor

    """

    if dims is None:
        dims = [i for i in range(input.dim() - 2, 0 if input.dim() == 4 else 1, -1)]  # H, W
    shift = [-input.size(dim) // 2 for dim in dims]
    return torch.roll(input, shift, dims)


def fftfreq(window_length: int,
            sample_spacing: float,
            *,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
            ) -> torch.Tensor:
    val = 1 / (window_length * sample_spacing)
    results = torch.empty(window_length, dtype=dtype, device=device)
    n = (window_length - 1) // 2 + 1
    results[:n] = torch.arange(0, n, dtype=dtype, device=device)
    results[n:] = torch.arange(-(window_length // 2), 0, dtype=dtype, device=device)
    return results * val
