import contextlib
from typing import Callable, Optional

import torch
from torch import Tensor, nn


def _svd(input: torch.Tensor
         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # torch.svd style
    U, S, Vh = torch.linalg.svd(input, full_matrices=False)
    V = Vh.transpose(-2, -1)
    return U, S, V


@torch.no_grad()
def _evaluate(model: nn.Module,
              data: tuple[Tensor, Tensor],
              criterion: Callable[[Tensor, Tensor], Tensor],
              auto_cast: bool
              ) -> float:
    # evaluate model with given data points using the criterion
    with (torch.cuda.amp.autocast() if auto_cast and torch.cuda.is_available() else contextlib.nullcontext()):
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
              dims: Optional[tuple[int, ...]] = None
              ) -> torch.Tensor:
    """ PyTorch version of np.fftshift

    Args:
        input: rFFTed Tensor of size [Bx]CxHxWx2
        dims:

    Returns: shifted tensor

    """

    return torch.fft.fftshift(input, dims)


def ifft_shift(input: torch.Tensor,
               dims: Optional[tuple[int, ...]] = None
               ) -> torch.Tensor:
    """ PyTorch version of np.ifftshift

    Args:
        input: rFFTed Tensor of size [Bx]CxHxWx2
        dims:

    Returns: shifted tensor

    """

    return torch.fft.ifftshift(input, dims)


def _rfft(self: Tensor,
          signal_ndim: int,
          normalized: bool = False,
          onesided: bool = True
          ) -> Tensor:
    # old-day's torch.rfft

    if signal_ndim > 4:
        raise RuntimeError("signal_ndim is expected to be 1, 2, 3.")

    m = torch.fft.rfftn if onesided else torch.fft.fftn
    dim = [-3, -2, -1][3 - signal_ndim:]
    return torch.view_as_real(m(self, dim=dim, norm="ortho" if normalized else None))


def _irfft(self: Tensor,
           signal_ndim: int,
           normalized: bool = False,
           onesided: bool = True,
           ) -> Tensor:
    # old-day's torch.irfft

    if signal_ndim > 4:
        raise RuntimeError("signal_ndim is expected to be 1, 2, 3.")
    if not torch.is_complex(self):
        self = torch.view_as_complex(self)

    m = torch.fft.irfftn if onesided else torch.fft.ifftn
    dim = [-3, -2, -1][3 - signal_ndim:]
    out = m(self, dim=dim, norm="ortho" if normalized else None)
    return out.real if torch.is_complex(out) else out
