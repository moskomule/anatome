from collections import OrderedDict
from importlib.metadata import version
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor, nn

AUTO_CAST = False
HAS_FFT_MODULE = (version("torch") >= "1.7.0")
if HAS_FFT_MODULE:
    import torch.fft


def use_auto_cast() -> None:
    """ Enable AMP autocast.
    """
    global AUTO_CAST
    AUTO_CAST = True


def _svd(input: torch.Tensor
         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # torch.svd style
    U, S, Vh = torch.linalg.svd(input, full_matrices=False)
    V = Vh.transpose(-2, -1)
    return U, S, V


@torch.no_grad()
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


def _rfft(self: Tensor,
          signal_ndim: int,
          normalized: bool = False,
          onesided: bool = True
          ) -> Tensor:
    # old-day's torch.rfft
    if not HAS_FFT_MODULE:
        return torch.rfft(self, signal_ndim, normalized, onesided)

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
    if not HAS_FFT_MODULE:
        return torch.irfft(self, signal_ndim, normalized, onesided)

    if signal_ndim > 4:
        raise RuntimeError("signal_ndim is expected to be 1, 2, 3.")
    if not torch.is_complex(self):
        self = torch.view_as_complex(self)

    m = torch.fft.irfftn if onesided else torch.fft.ifftn
    dim = [-3, -2, -1][3 - signal_ndim:]
    out = m(self, dim=dim, norm="ortho" if normalized else None)
    return out.real if torch.is_complex(out) else out


# - helper functions to compute dists/sims for actual networks and experiments

LayerIdentifier = Union[str, tuple[str]]

def _to_layer_order_dict(dists: list[float],
                         layer_names1: list[str],
                         layer_names2: list[str]) -> OrderedDict[LayerIdentifier, float]:
    """
    list([L, 1]) -> OrderedDict([L, 1])
    """
    assert len(layer_names1) == len(layer_names2), f'Expected same number of layers for comparing both nets but got: ' \
                                                   f'{(len(layer_names1))=}, {len(layer_names2)=}'
    # - put distances/sims for each layer
    dists_od: OrderedDict[LayerIdentifier, float] = OrderedDict()
    for layer_idx, layer1, layer2 in enumerate(zip(layer_names1, layer_names2)):
        dist: float = dists[layer_idx]
        layer_key: LayerIdentifier = layer1 if layer1 == layer2 else (layer1, layer2)
        dists_od[layer_key] = dist
    return dists_od

def _dists_per_layer_to_list(dists_per_layer: OrderedDict[LayerIdentifier, float]) -> list[float]:
    """
    OrderedDict([L, 1]) -> list([L, 1])
    """
    return [dist for _, dist in dists_per_layer]

def _dists_per_task_per_layer_to_list(dists_per_tasks_per_layer: list[OrderedDict[LayerIdentifier, float]]) -> list[list[float]]:
    """
    OrderedDict([B, L, 1]) -> list([B, L, 1])
    """
    B: int = len(dists_per_tasks_per_layer)
    _dists_per_tasks_per_layer: list[list[float]] = []
    for b in range(B):
        _dists_per_layer: list[float] = _dists_per_layer_to_list(dists_per_tasks_per_layer[b])
        _dists_per_tasks_per_layer.append(_dists_per_layer)
    return _dists_per_tasks_per_layer

def dist_data_set_per_layer(mdl1: nn.Module, mdl2: nn.Module,
                            X1: Tensor, X2: Tensor,
                            layer_names1: list[str], layer_names2: list[str],
                            dist_type: str,
                            downsample_method: Optional[str] = 'avg_pool',
                            downsample_size: Optional[int] = None,
                            effective_neuron_type: str = 'filter',
                            force_cpu: bool = False,
                            iters: int = 1) -> OrderedDict[LayerIdentifier, float]:
    """
    Given a pair of data sets (or batches, potentially the same one), compute the distance btw the models
        [M, C, H, W], [L] -> [L, 1] (as list like obj)
    assuming the list of layes are in the order of how you want to compute the comparison.
    E.g.
        - d(f_ml(X), A(f_ml, task)(X)), compute the distance between model and adapted model for same set of examples
        for the task/data set since X1=X2=X.
        - dv = d(f(X1), f(X2))
        - dv = d(f^*_1(X1), f^*_2(X2))

    :return: [L, 1] as a iterable (e.g. list of floats per layer) correponding to the distance for each layer.
    """
    from anatome.similarity import SimilarityHook
    assert len(layer_names1) == len(layer_names2), f'Expected same number of layers for comparing both nets but got: ' \
                                                   f'{(len(layer_names1))=}, {len(layer_names2)=}'
    # - create hooks for each layer name
    hooks1: list[SimilarityHook] = SimilarityHook.create_hooks(mdl1, layer_names1, dist_type, force_cpu)
    hooks2: list[SimilarityHook] = SimilarityHook.create_hooks(mdl2, layer_names2, dist_type, force_cpu)
    # - fill hooks with intermediate representations
    for _ in range(iters):  # might make sense to go through multiple is NN is stochastic e.g. BN, dropout layers
        # note we are avoiding torch.no_grad() because we don't want it to interfere with higher... could be future work to see if torch.no_grad() interferes with it.
        mdl1(X1)
        mdl2(X2)
    # - compute distances/sims for each layer
    dists_od: OrderedDict[LayerIdentifier, float] = OrderedDict()
    for layer_idx, layer1, layer2 in enumerate(zip(layer_names1, layer_names2)):
        hook1: SimilarityHook = hooks1[layer_idx]
        hook2: SimilarityHook = hooks2[layer_idx]
        # - get dist/sim for current models for current layer
        dist: float = hook1.distance(hook2, downsample_method=downsample_method, size=downsample_size,
                                     effective_neuron_type=effective_neuron_type)
        layer_key: LayerIdentifier = layer1 if layer1 == layer2 else (layer1, layer2)
        dists_od[layer_key] = dist
    return dists_od


def dist_batch_data_sets_for_all_layer(mdl1: nn.Module, mdl2: nn.Module,
                                       X1: Tensor, X2: Tensor,
                                       layer_names1: list[str], layer_names2: list[str],
                                       dist_type: str,
                                       downsample_method: Optional[str] = 'avg_pool',
                                       downsample_size: Optional[int] = None,
                                       effective_neuron_type: str = 'filter',
                                       force_cpu: bool = False,
                                       iters: int = 1) -> list[OrderedDict[LayerIdentifier, float]]:
    """
    Gets the distance for a batch (e.g meta-batch) of data sets/tasks and for each layer the distances between the nets:
        [B, M, C, H, W], [L] -> [B, L, 1] as nested iterable (e.g. list of list)
    Example use:
        - Computing d(f_ml(X), A(f_ml, t)(X)) so X1=X2=X.
        - Computing dv so some restricted cross product of the data sets in X1 and X2 (X1 != X2),
        ideally avoiding diagonal if dv.

    :return:
    """
    assert len(X1.size()) == 5, f'Data set does not 5 dims i.e. [B, M, C, H, W] instead got: {X1.size()=}'
    assert len(X2.size()) == 5, f'Data set does not 5 dims i.e. [B, M, C, H, W] instead got: {X2.size()=}'
    # - get distances per data sets per layers
    B: int = min(X1.size(0), X2.size(0))
    dists_entire_net_all_data_sets: list[OrderedDict[LayerIdentifier, float]] = []  # [B, L, 1]
    for b in range(B):
        # - [B, M, C, H, W] -> [M, C, H, W] (get a specific data set/batch)
        x1, x2 = X1[b], X2[b]
        # - compute for current pair of data sets (or tasks) the distance between models
        # [M, C, H, W], [L] -> [L, 1], really a list of floats of len [L]
        dist_for_data_set_b: OrderedDict[LayerIdentifier, float] = dist_data_set_per_layer(mdl1, mdl2, x1, x2,
                                                                                           layer_names1, layer_names2,
                                                                                           dist_type,
                                                                                           downsample_method,
                                                                                           downsample_size,
                                                                                           effective_neuron_type,
                                                                                           force_cpu, iters)
        # adding to [B, L, 1]
        dists_entire_net_all_data_sets.append(dist_for_data_set_b)
    # check effective size [B, L, 1]
    L: int = len(layer_names1)
    assert len(dists_entire_net_all_data_sets) == B
    assert len(dist_for_data_set_b[0]) == L
    # - [B, L, 1], return list of iters of distances per data sets (tasks) per layers
    return dists_entire_net_all_data_sets


def stats_distance_per_layer(mdl1: nn.Module, mdl2: nn.Module,
                             X1: Tensor, X2: Tensor,
                             layer_names1: list[str], layer_names2: list[str],
                             dist_type: str,
                             downsample_method: Optional[str] = 'avg_pool',
                             downsample_size: Optional[int] = None,
                             effective_neuron_type: str = 'filter',
                             force_cpu: bool = False,
                             iters: int = 1) -> tuple[OrderedDict[LayerIdentifier, float]]:
    """
    Compute the overall stats (mean & std) of distances per layer
        [B, M, C, H, W] -> (mu_per_layer, std_per_layer) pair of size [L, 1]
    for the given B number of data sets/tasks of size [M, C, H, W].

    Note:
        - this might be confusing because X1 != X2. The most intuitive case is when X1=X2=X so that we are computing
        the distance for each task (but the tasks correspond to each other).
    :return:
    """
    from uutils.torch_uu import tensorify
    # - [B, L, 1], get distances per data sets (tasks) per layer
    distances_per_data_sets_per_layer: list[OrderedDict[LayerIdentifier, float]] = dist_batch_data_sets_for_all_layer(mdl1, mdl2, X1, X2,
                                                                                              layer_names1,
                                                                                              layer_names2, dist_type,
                                                                                              downsample_method,
                                                                                              downsample_size,
                                                                                              effective_neuron_type,
                                                                                              force_cpu, iters)
    # list(OrderDict([B, L, 1])) -> list([B, L, 1])
    _distances_per_data_sets_per_layer: list[list[float]] = _dists_per_task_per_layer_to_list(distances_per_data_sets_per_layer)
    _distances_per_data_sets_per_layer: Tensor = tensorify(_distances_per_data_sets_per_layer)
    # - [B, L, 1] -> [L, 1], get the avg distance/sim for each layer (and std)
    L: int = len(_distances_per_data_sets_per_layer[0])
    means_per_layer: Tensor = _distances_per_data_sets_per_layer.mean(dim=0)
    stds_per_layer: Tensor = _distances_per_data_sets_per_layer.std(dim=0)
    assert means_per_layer.size() == torch.Size([L, 1])
    assert stds_per_layer.size() == torch.Size([L, 1])
    # [L, 1] -> OrderDict([L, 1])
    mean_distance_per_layer: OrderedDict[LayerIdentifier, float] = []
    std_distance_per_layer: OrderedDict[LayerIdentifier, float] = []
    for l_idx in range(L):




def stats_distance_entire_net(mdl1: nn.Module, mdl2: nn.Module,
                              X1: Tensor, X2: Tensor,
                              layer_names1: list[str], layer_names2: list[str],
                              dist_type: str,
                              downsample_method: Optional[str] = 'avg_pool',
                              downsample_size: Optional[int] = None,
                              effective_neuron_type: str = 'filter',
                              force_cpu: bool = False,
                              iters: int = 1) -> tuple[float]:
    """
    Compute the overall stats of distances per layer
        [B, M, C, H, W] -> pair of [1, 1]
    :return:
    """
    pass

# - tests

# def multiple_hooks_test():
#     """
#     """
#     from collections import OrderedDict
#     from copy import deepcopy
#
#     import torch
#     import torch.nn as nn
#
#     from uutils.torch_uu import cxa_sim, approx_equal
#     from uutils.torch_uu.models import get_single_conv_model
#     import anatome
#     print(f'from import: {anatome=}')
#
#     # - very simple sanity check
#     Cin: int = 3
#     num_out_filters: int = 8
#     conv_layer: nn.Module = get_single_conv_model(in_channels=3, num_out_filters=num_out_filters)
#     mdl1: nn.Module = nn.Sequential(OrderedDict([('conv1', conv_layer)]))
#     mdl2: nn.Module = deepcopy(mdl1)
#     layer_name = 'conv1'
#
#     # - ends up comparing two matrices of size [B, Dout], on same data, on same model
#     B: int = 4
#     C, H, W = Cin, 64, 64
#     # downsample_size = None
#     cxa_dist_type = 'svcca'
#     cxa_dist_type = 'pwcca'
#     X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, C, H, W))
#
#     # - compute sim for NO downsample: so layer matrix is []
#     downsample_size = None
#     sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=downsample_size, iters=1,
#                          cxa_dist_type=cxa_dist_type)
#     print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
#     assert (approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'
#     # -- multiple hooks
#
#     pass
