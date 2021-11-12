from __future__ import annotations

from functools import partial
from typing import Callable, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from .utils import _irfft, _rfft, _svd


def _zero_mean(input: Tensor,
               dim: int
               ) -> Tensor:
    return input - input.mean(dim=dim, keepdim=True)


def _divide_by_max(input: Tensor
                   ) -> Tensor:
    return input / input.abs().max()


def _check_shape_equal(x: Tensor,
                       y: Tensor,
                       dim: int
                       ):
    if x.size(dim) != y.size(dim):
        raise ValueError(f'x.size({dim}) == y.size({dim}) is expected, but got {x.size(dim)=}, {y.size(dim)=} instead.')


def cca_by_svd(x: Tensor,
               y: Tensor
               ) -> tuple[Tensor, Tensor, Tensor]:
    """ CCA using only SVD.
    For more details, check Press 2011 "Canonical Correlation Clarified by Singular Value Decomposition"

    Args:
        x: input tensor of Shape DxH
        y: input tensor of shape DxW

    Returns: x-side coefficients, y-side coefficients, diagonal

    """

    # torch.svd(x)[1] is vector
    u_1, s_1, v_1 = _svd(x)
    u_2, s_2, v_2 = _svd(y)
    uu = u_1.t() @ u_2
    u, diag, v = _svd(uu)
    # a @ (1 / s_1).diag() @ u, without creating s_1.diag()
    a = v_1 @ (1 / s_1[:, None] * u)
    b = v_2 @ (1 / s_2[:, None] * v)
    return a, b, diag


def cca_by_qr(x: Tensor,
              y: Tensor
              ) -> tuple[Tensor, Tensor, Tensor]:
    """ CCA using QR and SVD.
    For more details, check Press 2011 "Canonical Correlation Clarified by Singular Value Decomposition"

    Args:
        x: input tensor of Shape DxH
        y: input tensor of shape DxW

    Returns: x-side coefficients, y-side coefficients, diagonal

    """

    q_1, r_1 = torch.linalg.qr(x)
    q_2, r_2 = torch.linalg.qr(y)
    qq = q_1.t() @ q_2
    u, diag, v = _svd(qq)
    # a = r_1.inverse() @ u, but it is faster and more numerically stable
    a = torch.linalg.solve(r_1, u)
    b = torch.linalg.solve(r_2, v)
    return a, b, diag


def cca(x: Tensor,
        y: Tensor,
        backend: str
        ) -> tuple[Tensor, Tensor, Tensor]:
    """ Compute CCA, Canonical Correlation Analysis

    Args:
        x: input tensor of Shape DxH
        y: input tensor of Shape DxW
        backend: svd or qr

    Returns: x-side coefficients, y-side coefficients, diagonal

    """

    _check_shape_equal(x, y, 0)

    if x.size(0) < x.size(1):
        raise ValueError(f'x.size(0) >= x.size(1) is expected, but got {x.size()=}.')

    if y.size(0) < y.size(1):
        raise ValueError(f'y.size(0) >= y.size(1) is expected, but got {y.size()=}.')

    if backend not in ('svd', 'qr'):
        raise ValueError(f'backend is svd or qr, but got {backend}')

    x = _divide_by_max(_zero_mean(x, dim=0))
    y = _divide_by_max(_zero_mean(y, dim=0))
    return cca_by_svd(x, y) if backend == 'svd' else cca_by_qr(x, y)


def _svd_reduction(input: Tensor,
                   accept_rate: float
                   ) -> Tensor:
    left, diag, right = _svd(input)
    full = diag.abs().sum()
    ratio = diag.abs().cumsum(dim=0) / full
    num = torch.where(ratio < accept_rate,
                      input.new_ones(1, dtype=torch.long),
                      input.new_zeros(1, dtype=torch.long)
                      ).sum()
    return input @ right[:, : num]


def svcca_distance(x: Tensor,
                   y: Tensor,
                   accept_rate: float,
                   backend: str
                   ) -> Tensor:
    """ Singular Vector CCA proposed in Raghu et al. 2017.

    Args:
        x: input tensor of Shape DxH, where D>H
        y: input tensor of Shape DxW, where D>H
        accept_rate: 0.99
        backend: svd or qr

    Returns:

    """

    x = _svd_reduction(x, accept_rate)
    y = _svd_reduction(y, accept_rate)
    div = min(x.size(1), y.size(1))
    a, b, diag = cca(x, y, backend)
    return 1 - diag.sum() / div


def pwcca_distance(x: Tensor,
                   y: Tensor,
                   backend: str
                   ) -> Tensor:
    """ Projection Weighted CCA proposed in Marcos et al. 2018.

    Args:
        x: input tensor of Shape DxH, where D>H
        y: input tensor of Shape DxW, where D>H
        backend: svd or qr

    Returns:

    """

    a, b, diag = cca(x, y, backend)
    alpha = (x @ a).abs_().sum(dim=0)
    alpha /= alpha.sum()
    return 1 - alpha @ diag


def _debiased_dot_product_similarity(z: Tensor,
                                     sum_row_x: Tensor,
                                     sum_row_y: Tensor,
                                     sq_norm_x: Tensor,
                                     sq_norm_y: Tensor,
                                     size: int
                                     ) -> Tensor:
    return (z
            - size / (size - 2) * (sum_row_x @ sum_row_y)
            + sq_norm_x * sq_norm_y / ((size - 1) * (size - 2)))


def linear_cka_distance(x: Tensor,
                        y: Tensor,
                        reduce_bias: bool
                        ) -> Tensor:
    """ Linear CKA used in Kornblith et al. 19
    
    Args:
        x: input tensor of Shape DxH
        y: input tensor of Shape DxW
        reduce_bias: debias CKA estimator, which might be helpful when D is limited

    Returns:

    """

    _check_shape_equal(x, y, 0)

    x = _divide_by_max(_zero_mean(x, dim=0))
    y = _divide_by_max(_zero_mean(y, dim=0))
    dot_prod = (y.t() @ x).norm('fro').pow(2)
    norm_x = (x.t() @ x).norm('fro')
    norm_y = (y.t() @ y).norm('fro')

    if reduce_bias:
        size = x.size(0)
        # (x @ x.t()).diag()
        sum_row_x = torch.einsum('ij,ij->i', x, x)
        sum_row_y = torch.einsum('ij,ij->i', y, y)
        sq_norm_x = sum_row_x.sum()
        sq_norm_y = sum_row_y.sum()
        dot_prod = _debiased_dot_product_similarity(dot_prod, sum_row_x, sum_row_y, sq_norm_x, sq_norm_y, size)
        norm_x = _debiased_dot_product_similarity(norm_x.pow_(2), sum_row_x, sum_row_y, sq_norm_x, sq_norm_y, size)
        norm_y = _debiased_dot_product_similarity(norm_y.pow_(2), sum_row_x, sum_row_y, sq_norm_x, sq_norm_y, size)
    return 1 - dot_prod / (norm_x * norm_y)


def orthogonal_procrustes_distance(x: Tensor,
                                   y: Tensor,
                                   ) -> Tensor:
    """ Orthogonal Procrustes distance used in Ding+21

    Args:
        x: input tensor of Shape DxH
        y: input tensor of Shape DxW

    Returns:

    """
    _check_shape_equal(x, y, 0)

    frobenius_norm = partial(torch.linalg.norm, ord="fro")
    nuclear_norm = partial(torch.linalg.norm, ord="nuc")

    x = _divide_by_max(_zero_mean(x, dim=0))
    x /= frobenius_norm(x)
    y = _divide_by_max(_zero_mean(y, dim=0))
    y /= frobenius_norm(y)
    # frobenius_norm(x) = 1, frobenius_norm(y) = 1
    # 0.5*d_proc(x, y)
    return 1 - nuclear_norm(x.t() @ y)


class Distance(object):
    """ Module to measure distance between `model1` and `model2`

    Args:
        method: Method to compute distance. 'pwcca' by default.
        model1_names: Names of modules of `model1` to be used. If None (default), all names are used.
        model2_names: Names of modules of `model2` to be used. If None (default), all names are used.
        model1_leaf_modules: Modules of model1 to be considered as single nodes (see https://pytorch.org/blog/FX-feature-extraction-torchvision/).
        model2_leaf_modules: Modules of model2 to be considered as single nodes (see https://pytorch.org/blog/FX-feature-extraction-torchvision/).
        train_mode: If True, models' `train_model` is used, otherwise `eval_mode`. False by default.
    """

    _supported_dims = (2, 4)
    _default_backends = {'pwcca': partial(pwcca_distance, backend='svd'),
                         'svcca': partial(svcca_distance, accept_rate=0.99, backend='svd'),
                         'lincka': partial(linear_cka_distance, reduce_bias=False),
                         'opd': orthogonal_procrustes_distance}

    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 method: str | Callable = 'pwcca',
                 model1_names: str | list[str] = None,
                 model2_names: str | list[str] = None,
                 model1_leaf_modules: list[nn.Module] = None,
                 model2_leaf_modules: list[nn.Module] = None,
                 train_mode: bool = False
                 ):

        dp_ddp = (nn.DataParallel, nn.parallel.DistributedDataParallel)
        if isinstance(model1, dp_ddp) or isinstance(model2, dp_ddp):
            raise RuntimeWarning('model is nn.DataParallel or nn.DistributedDataParallel. '
                                 'SimilarityHook may causes unexpected behavior.')
        if isinstance(method, str):
            method = self._default_backends[method]
        self.distance_func = method
        self.model1 = model1
        self.model2 = model2
        self.extractor1 = create_feature_extractor(model1, self.convert_names(model1, model1_names,
                                                                              model1_leaf_modules, train_mode))
        self.extractor2 = create_feature_extractor(model2, self.convert_names(model2, model2_names,
                                                                              model2_leaf_modules, train_mode))
        self._model1_tensors: dict[str, torch.Tensor] = None
        self._model2_tensors: dict[str, torch.Tensor] = None

    def available_names(self,
                        model1_leaf_modules: list[nn.Module] = None,
                        model2_leaf_modules: list[nn.Module] = None,
                        train_mode: bool = False
                        ):
        return {'model1': self.convert_names(self.model1, None, model1_leaf_modules, train_mode),
                'model2': self.convert_names(self.model2, None, model2_leaf_modules, train_mode)}

    @staticmethod
    def convert_names(model: nn.Module,
                      names: str | list[str],
                      leaf_modules: list[nn.Module],
                      train_mode: bool
                      ) -> list[str]:
        # a helper function
        if isinstance(names, str):
            names = [names]
        tracer_kwargs = {}
        if leaf_modules is not None:
            tracer_kwargs['leaf_modules'] = leaf_modules

        _names = get_graph_node_names(model, tracer_kwargs=tracer_kwargs)
        _names = _names[0] if train_mode else _names[1]
        _names = _names[1:]  # because the first element is input

        if names is None:
            names = _names
        else:
            if not (set(names) <= set(_names)):
                diff = set(names) - set(_names)
                raise RuntimeError(f'Unknown names: {list(diff)}')

        return names

    def forward(self,
                data
                ) -> None:
        """ Forward pass of models. Used to store intermediate features.

        Args:
            data: input data to models

        """
        self._model1_tensors = self.extractor1(data)
        self._model2_tensors = self.extractor1(data)

    def between(self,
                name1: str,
                name2: str,
                size: int | tuple[int, int] = None,
                downsample_method: Literal['avg_pool', 'dft'] = 'avg_pool'
                ) -> torch.Tensor:
        """ Compute distance between modules corresponding to name1 and name2.

        Args:
            name1: Name of a module of `model1`
            name2: Name of a module of `model2`
            size: Size for downsampling if necessary. If size's type is int, both features of name1 and name2 are
            reshaped to (size, size). If size's type is tuple[int, int], features are reshaped to (size[0], size[0]) and
            (size[1], size[1]). If size is None (default), no downsampling is applied.
            downsample_method: Downsampling method: 'avg_pool' for average pooling and 'dft' for discrete
            Fourier transform

        Returns: Distance in tensor.

        """
        tensor1 = self._model1_tensors[name1]
        tensor2 = self._model2_tensors[name2]
        if tensor1.dim() not in self._supported_dims:
            raise RuntimeError(f'Supported dimensions are ={self._supported_dims}, but got {tensor1.dim()}')
        if tensor2.dim() not in self._supported_dims:
            raise RuntimeError(f'Supported dimensions are ={self._supported_dims}, but got {tensor2.dim()}')

        if size is not None:
            if isinstance(size, int):
                size = (size, size)

            def downsample_if_necessary(input, s):
                if input.dim() == 4:
                    input = self._downsample_4d(input, s, downsample_method)
                return input

            tensor1 = downsample_if_necessary(tensor1, size[0])
            tensor2 = downsample_if_necessary(tensor2, size[1])

        def reshape_if_4d(input):
            if input.dim() == 4:
                # see https://arxiv.org/abs/1706.05806's P5.
                if name1 == name2:  # same layer comparisons -> Cx(BHW)
                    input = input.permute(1, 0, 2, 3).flatten(1)
                else:  # different layer comparisons -> Bx(CHW)
                    input = input.flatten(1)
            return input

        tensor1 = reshape_if_4d(tensor1)
        tensor2 = reshape_if_4d(tensor2)

        return self.distance_func(tensor1, tensor2)

    @staticmethod
    def _downsample_4d(input: Tensor,
                       size: int,
                       backend: Literal['avg_pool', 'dft']
                       ) -> Tensor:
        if input.dim() != 4:
            raise RuntimeError(f'input is expected to be 4D tensor, but got {input.dim()=}.')

        # todo: what if channel-last?
        b, c, h, w = input.size()

        if (size, size) == (h, w):
            return input

        if (size, size) > (h, w):
            raise RuntimeError(f'size ({size}) is expected to be smaller than h or w, but got {h=}, {w=}.')

        if backend not in ('avg_pool', 'dft'):
            raise RuntimeError(f'backend is expected to be avg_pool or dft, but got {backend=}.')

        if backend == 'avg_pool':
            return F.adaptive_avg_pool2d(input, (size, size))

        # almost PyTorch implant of
        # https://github.com/google/svcca/blob/master/dft_ccas.py
        if input.size(2) != input.size(3):
            raise RuntimeError('width and height of input needs to be equal')
        h = input.size(2)
        input_fft = _rfft(input, 2, normalized=True, onesided=False)
        freqs = torch.fft.fftfreq(h, 1 / h, device=input.device)
        idx = (freqs >= -size / 2) & (freqs < size / 2)
        # BxCxHxWx2 -> BxCxhxwx2
        input_fft = input_fft[..., idx, :][..., idx, :, :]
        input = _irfft(input_fft, 2, normalized=True, onesided=False)
        return input
