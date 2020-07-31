from typing import Tuple, Callable, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from tqdm import tqdm

from .utils import _evaluate


def fft_shift(input: torch.Tensor
              ) -> torch.Tensor:
    """ PyTorch version of np.fftshift

    Args:
        input: rFFTed Tensor of size [Bx]CxHxWx2

    Returns: shifted tensor

    """

    dims = [i for i in range(1 if input.dim() == 4 else 2, input.dim() - 1)]  # H, W
    shift = [input.size(dim) // 2 for dim in dims]
    return torch.roll(input, shift, dims)


def ifft_shift(input: torch.Tensor
               ) -> torch.Tensor:
    """ PyTorch version of np.ifftshift

    Args:
        input: rFFTed Tensor of size [Bx]CxHxWx2

    Returns: shifted tensor

    """

    dims = [i for i in range(input.dim() - 2, 0 if input.dim() == 4 else 1, -1)]  # H, W
    shift = [-input.size(dim) // 2 for dim in dims]
    return torch.roll(input, shift, dims)


def add_fourier_noise(idx: Tuple[int, int],
                      images: Tensor,
                      norm: float,
                      size: Optional[Tuple[int, int]] = None
                      ) -> Tensor:
    """ Add Fourier noise

    Args:
        idx: index to be used
        images: original images
        norm: norm of additive noise
        size:

    Returns: images with Fourier noise

    """

    images = images.clone()

    if size is None:
        _, _, h, w = images.size()
    else:
        h, w = size
    noise = images.new_zeros(1, h, w, 2)
    noise[:, idx[0], idx[1]] = 1
    noise[:, h - 1 - idx[0], w - 1 - idx[0]] = 1
    recon = ifft_shift(noise).irfft(2, normalized=True, onesided=False).unsqueeze(0)
    if size is not None:
        recon = F.interpolate(recon, images.shape[2:])
    images.add_(recon.div_(recon.norm(p=2)).mul_(norm))
    return images


@torch.no_grad()
def fourier_map(model: nn.Module,
                data: Tuple[Tensor, Tensor],
                criterion: Callable[[Tensor, Tensor], Tensor],
                norm: float,
                fourier_map_size: Optional[Tuple[int, int]] = None
                ) -> Tensor:
    input, target = data
    if fourier_map_size is None:
        _, _, h, w = input.size()
    else:
        h, w = fourier_map_size
    map = torch.zeros(h, w)
    for u_i, l_i in tqdm(zip(torch.triu_indices(h, w).t(),
                             torch.tril_indices(h, w).t()),
                         ncols=80):
        input = add_fourier_noise(u_i, input, norm, fourier_map_size)
        loss = _evaluate(model, (input, target), criterion)
        map[u_i] = loss
        map[l_i] = loss
    return map
