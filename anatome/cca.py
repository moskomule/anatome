from __future__ import annotations

from functools import partial
from typing import Tuple, Optional, Callable

import torch
from torch import nn
from torch.nn import functional as F


def _zero_mean(input: torch.Tensor,
               dim: int
               ) -> torch.Tensor:
    return input - input.mean(dim=dim, keepdim=True)


def cca_by_svd(x: torch.Tensor,
               y: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ CCA using only SVD.
    For more details, check Press 2011 "Cannonical Correlation Clarified by Singular Value Decomposition"

    Args:
        x: input tensor of Shape DxH
        y: input tensor of shape DxW

    Returns: x-side coefficients, y-side coefficients, diagonal

    """

    # torch.svd(x)[1] is vector
    u_1, s_1, v_1 = torch.svd(x)
    u_2, s_2, v_2 = torch.svd(y)
    uu = u_1.t() @ u_2
    u, diag, v = torch.svd(uu)
    # v @ s.diag() = v * s.view(-1, 1), but much faster
    a = (v_1 * s_1.reciprocal_().unsqueeze_(0)) @ u
    b = (v_2 * s_2.reciprocal_().unsqueeze_(0)) @ v
    return a, b, diag


def cca_by_qr(x: torch.Tensor,
              y: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ CCA using QR and SVD.
    For more details, check Press 2011 "Cannonical Correlation Clarified by Singular Value Decomposition"

    Args:
        x: input tensor of Shape DxH
        y: input tensor of shape DxW

    Returns: x-side coefficients, y-side coefficients, diagonal

    """

    q_1, r_1 = torch.qr(x)
    q_2, r_2 = torch.qr(y)
    qq = q_1.t() @ q_2
    u, diag, v = torch.svd(qq)
    a = torch.inverse(r_1) @ u
    b = torch.inverse(r_2) @ v
    return a, b, diag


def cca(x: torch.Tensor,
        y: torch.Tensor,
        backend: str
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Compute CCA

    Args:
        x: input tensor of Shape DxH
        y: input tensor of Shape DxW
        backend: svd or qr

    Returns: x-side coefficients, y-side coefficients, diagonal

    """

    if x.size(0) != y.size(0):
        raise ValueError(f'x.size(0) == y.size(0) is expected, but got {x.size(0)=}, {y.size(0)=} instead.')

    if x.size(0) < x.size(1):
        raise ValueError(f'x.size(0) >= x.size(1) is expected, but got {x.size()=}.')

    if y.size(0) < y.size(1):
        raise ValueError(f'y.size(0) >= y.size(1) is expected, but got {y.size()=}.')

    if backend not in ('svd', 'qr'):
        raise ValueError(f'backend is svd or qr, but got {backend}')

    x = _zero_mean(x, dim=0)
    y = _zero_mean(y, dim=0)
    return cca_by_svd(x, y) if backend == 'svd' else cca_by_qr(x, y)


def _svd_reduction(input: torch.Tensor,
                   accept_rate: float
                   ) -> torch.Tensor:
    left, diag, right = torch.svd(input)
    full = diag.abs().sum()
    ratio = diag.abs().cumsum(dim=0) / full
    num = torch.where(ratio < accept_rate,
                      input.new_ones(1, dtype=torch.long),
                      input.new_zeros(1, dtype=torch.long)
                      ).sum()
    return input @ right[:, : num]


def svcca_distance(x: torch.Tensor,
                   y: torch.Tensor,
                   accept_rate: float,
                   backend: str
                   ) -> torch.Tensor:
    """ Singular Vector CCA proposed in Raghu et al. 2017.

    Args:
        x: input tensor of Shape DxH
        y: input tensor of Shape DxW
        accept_rate: 0.99
        backend: svd or qr

    Returns:

    """

    x = _svd_reduction(x, accept_rate)
    y = _svd_reduction(y, accept_rate)
    div = min(x.size(1), y.size(1))
    a, b, diag = cca(x, y, backend)
    return 1 - diag.sum() / div


def pwcca_distance(x: torch.Tensor,
                   y: torch.Tensor,
                   backend: str
                   ) -> torch.Tensor:
    """ Projection Weighted CCA proposed in Marcos et al. 2018.

    Args:
        x: input tensor of Shape DxH
        y: input tensor of Shape DxW
        backend: svd or qr

    Returns:

    """

    a, b, diag = cca(x, y, backend)
    alpha = (x @ a).abs_().sum(dim=0)
    alpha /= alpha.sum()
    return 1 - alpha @ diag


class CCAHook(object):
    """ Hook to compute CCA distance between modules in given models ::

        model = resnet18()
        hook1 = CCAHook(model, "layer3.0.conv1")
        hook2 = CCAHook(model, "layer3.0.conv2")
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                model(torch.randn(120, 3, 224, 224))
        hook1.distance(hook2, size=8)

    Args:
        model: Model
        module_name: Name of module appeared in `model.named_modules()`
        cca_distance: the method to compute CCA distance. By default, PWCCA is used.
        'pwcca', 'svcca' or partial functions such as `partial(pwcca_distance, backend='qr')` are expected.
        force_cpu: Force computing CCA on CPUs. In some cases, CCA computation is faster on CPUs than on GPUs.
    """

    _supported_dim = (2, 4)
    _default_backends = {'pwcca': partial(pwcca_distance, backend='svd'),
                         'svcca': partial(svcca_distance, accept_rate=0.99, backend='svd')}

    def __init__(self,
                 model: nn.Module,
                 module_name: str,
                 cca_distance: Optional[str or Callable] = None,
                 force_cpu: bool = False,
                 ) -> None:

        self.model = model
        self.module = {k: v for k, v in self.model.named_modules()}[module_name]
        if cca_distance is None or isinstance(cca_distance, str):
            cca_distance = self._default_backends[cca_distance or 'pwcca']
        self.cca_function = cca_distance

        self.force_cpu = force_cpu
        if self.force_cpu:
            # fully utilize CPUs available
            from multiprocessing import cpu_count
            torch.set_num_threads(cpu_count())

        self._hooked_tensors = None
        self.register_hook()

    def clear(self
              ) -> None:
        self._hooked_tensors = None

    @property
    def hooked_tensors(self
                       ) -> torch.Tensor:
        if self._hooked_tensors is None:
            raise RuntimeError('Run model in advance')
        return self._hooked_tensors

    def register_hook(self
                      ) -> None:
        def hook(*args):
            output = args[2]
            if output.dim() not in self._supported_dim:
                raise RuntimeError(f'CCAHook currently supports tensors of dimension {self._supported_dim}, '
                                   f'but got {output.dim()} instead.')

            if self.force_cpu:
                output = output.cpu()

            if self._hooked_tensors is None:
                self._hooked_tensors = output
            else:
                self._hooked_tensors = torch.cat([self._hooked_tensors, output], dim=0)

        self.module.register_forward_hook(hook)

    def distance(self,
                 other: CCAHook,
                 *,
                 downsample_method: Optional[str] = 'avg_pool',
                 size: Optional[int] = None,
                 ) -> float:
        """ Compute CCA distance between self and other.

        Args:
            other: Another hook
            downsample_method: method for
            size:

        Returns:

        """

        self_tensor = self.hooked_tensors
        other_tensor = other.hooked_tensors
        if self_tensor.size(0) != other_tensor.size(0):
            raise RuntimeError('0th dimension of hooked tensors are different')
        if self_tensor.dim() != other_tensor.dim():
            raise RuntimeError('Dimensions of hooked tensors need to be same, '
                               f'but got {self_tensor.dim()=} and {other_tensor.dim()=}')

        if self_tensor.dim() == 2:
            return self.cca_function(self_tensor, other_tensor).item()
        else:
            if size is not None:
                downsample_method = downsample_method or 'avg_pool'
                self_tensor = self._downsample_4d(self_tensor, size, downsample_method)
                other_tensor = self._downsample_4d(other_tensor, size, downsample_method)
            return torch.stack([self.cca_function(s, o)
                                for s, o in zip(self_tensor.unbind(), other_tensor.unbind())
                                ]
                               ).mean()

    @staticmethod
    def _downsample_4d(input: torch.Tensor,
                       size: int,
                       backend: str
                       ) -> torch.Tensor:
        """ Downsample 4D tensor of BxCxHxD to 3D tensor of {size^2}xBxC
        """

        if input.dim() != 4:
            raise RuntimeError(f'input is expected to be 4D tensor, but got {input.dim()=}.')

        # todo: what if channel-last?
        b, c, h, w = input.size()

        if (size, size) > (h, w):
            raise RuntimeError(f'size is expected to be smaller than h or w, but got {h=}, {w=}.')

        if backend not in ('avg_pool', 'dft'):
            raise RuntimeError(f'backend is expected to be avg_pool or dft, but got {backend=}.')

        if backend == 'avg_pool':
            input = F.adaptive_avg_pool2d(input, (size, size))
            input = input.view(b, c, -1).permute(2, 0, 1)
        else:
            # todo: check SVCCA paper
            pass

        return input
