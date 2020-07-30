from typing import Tuple
import torch


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
        raise ValueError(f'x.size(0) >= x.size(1) is expected, but got {x.size()=}')

    if y.size(0) < y.size(1):
        raise ValueError(f'y.size(0) >= y.size(1) is expected, but got {y.size()=}')

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
    """ Projection Weighted CCA proposed in Macros et al. 2018.

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
