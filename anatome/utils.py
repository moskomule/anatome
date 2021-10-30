import time
from collections import OrderedDict
from importlib.metadata import version
from pprint import pprint
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


def _clear_hooks(hooks: list):
    for hook in hooks:
        hook.clear()


def remove_hook(mdl: nn.Module, hook):
    """
    ref: https://github.com/pytorch/pytorch/issues/5037
    """
    handle = mdl.register_forward_hook(hook)
    handle.remove()


def remove_hooks(mdl: nn.Module, hooks: list):
    """
    ref:
        - https://github.com/pytorch/pytorch/issues/5037
        - https://discuss.pytorch.org/t/how-to-remove-multiple-hooks/135442
    """
    for hook in hooks:
        remove_hook(mdl, hook)


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
    return [dist for _, dist in dists_per_layer.items()]


def _dists_per_task_per_layer_to_list(dists_per_tasks_per_layer: list[OrderedDict[LayerIdentifier, float]]) -> list[
    list[float]]:
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
                            iters: int = 1,
                            metrics_as_dist: bool = True,
                            ) -> OrderedDict[LayerIdentifier, float]:
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
    for layer_idx, (layer1, layer2) in enumerate(zip(layer_names1, layer_names2)):
        hook1: SimilarityHook = hooks1[layer_idx]
        hook2: SimilarityHook = hooks2[layer_idx]
        # - get dist/sim for current models for current layer
        dist: float = hook1.distance(hook2, downsample_method=downsample_method, size=downsample_size,
                                     effective_neuron_type=effective_neuron_type)
        layer_key: LayerIdentifier = layer1 if layer1 == layer2 else (layer1, layer2)
        dist = dist if metrics_as_dist else 1.0 - dist
        dists_od[layer_key] = dist
    _clear_hooks(hooks1)
    _clear_hooks(hooks2)
    return dists_od


def dist_batch_data_sets_for_all_layer(mdl1: nn.Module, mdl2: nn.Module,
                                       X1: Tensor, X2: Tensor,
                                       layer_names1: list[str], layer_names2: list[str],
                                       dist_type: str,
                                       downsample_method: Optional[str] = 'avg_pool',
                                       downsample_size: Optional[int] = None,
                                       effective_neuron_type: str = 'filter',
                                       force_cpu: bool = False,
                                       iters: int = 1,
                                       metrics_as_dist: bool = True) -> list[OrderedDict[LayerIdentifier, float]]:
    """
    Gets the distance for a batch (e.g meta-batch) of data sets/tasks and for each layer the distances between the nets:
        [B, M, C, H, W], [L] -> list(OrderDict([B, L, 1]))
    Example use:
        - Computing d(f_ml(X), A(f_ml, t)(X)) so X1=X2=X.
        - Computing dv so some restricted cross product of the data sets in X1 and X2 (X1 != X2),
        ideally avoiding diagonal if dv.

    :return:
    """
    assert len(X1.size()) == 5, f'Data set does not 5 dims i.e. [B, M, C, H, W] instead got: {X1.size()=}'
    assert len(X2.size()) == 5, f'Data set does not 5 dims i.e. [B, M, C, H, W] instead got: {X2.size()=}'
    # - get distances per data sets/tasks per layers: [B, M, C, H, W], [L] -> [B, L, 1]
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
                                                                                           force_cpu, iters,
                                                                                           metrics_as_dist
                                                                                           )
        # adding to [B, L, 1]
        dists_entire_net_all_data_sets.append(dist_for_data_set_b)
    # check effective size [B, L, 1]
    L: int = len(layer_names1)
    assert len(dists_entire_net_all_data_sets) == B
    assert len(dists_entire_net_all_data_sets[0]) == L
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
                             iters: int = 1,
                             metrics_as_dist: bool = True
                             ) -> tuple[OrderedDict[LayerIdentifier, float]]:
    """
    Compute the overall stats (mean & std) of distances per layer
        [B, M, C, H, W] -> [L, 1]*2 (mus, stds) per layer
    for the given B number of data sets/tasks of size [M, C, H, W].

    Note:
        - this might be confusing because X1 != X2. The most intuitive case is when X1=X2=X so that we are computing
        the distance for each task (but the tasks correspond to each other).
    :return:
    """
    from uutils.torch_uu import tensorify
    assert len(layer_names1) == len(layer_names2)
    # - [B, L, 1], get distances per data sets (tasks) per layer
    distances_per_data_sets_per_layer: list[OrderedDict[LayerIdentifier, float]] = dist_batch_data_sets_for_all_layer(
        mdl1, mdl2, X1, X2,
        layer_names1,
        layer_names2, dist_type,
        downsample_method,
        downsample_size,
        effective_neuron_type,
        force_cpu, iters,
        metrics_as_dist
        )
    # list(OrderDict([B, L, 1])) -> list([B, L, 1])
    _distances_per_data_sets_per_layer: list[list[float]] = _dists_per_task_per_layer_to_list(
        distances_per_data_sets_per_layer)
    _distances_per_data_sets_per_layer: Tensor = tensorify(_distances_per_data_sets_per_layer)
    # - [B, L, 1] -> [L, 1], get the avg distance/sim for each layer (and std)
    means_per_layer: Tensor = _distances_per_data_sets_per_layer.mean(dim=0)
    stds_per_layer: Tensor = _distances_per_data_sets_per_layer.std(dim=0)
    L: int = len(_distances_per_data_sets_per_layer[0])
    assert means_per_layer.size() == torch.Size([L])
    assert stds_per_layer.size() == torch.Size([L])
    # [L, 1] -> OrderDict([L])
    layer_ids: list[LayerIdentifier] = list(distances_per_data_sets_per_layer[0].keys())
    assert len(layer_ids) == L == len(layer_names1)
    mean_distance_per_layer: OrderedDict[LayerIdentifier, float] = OrderedDict()
    std_distance_per_layer: OrderedDict[LayerIdentifier, float] = OrderedDict()
    for l_idx in range(L):
        layer_id: LayerIdentifier = layer_ids[l_idx]
        mu: float = means_per_layer[l_idx].item()
        std: float = stds_per_layer[l_idx].item()
        mean_distance_per_layer[layer_id] = mu
        std_distance_per_layer[layer_id] = std
    return mean_distance_per_layer, std_distance_per_layer


def stats_distance_entire_net(mdl1: nn.Module, mdl2: nn.Module,
                              X1: Tensor, X2: Tensor,
                              layer_names1: list[str], layer_names2: list[str],
                              dist_type: str,
                              downsample_method: Optional[str] = 'avg_pool',
                              downsample_size: Optional[int] = None,
                              effective_neuron_type: str = 'filter',
                              force_cpu: bool = False,
                              iters: int = 1,
                              metrics_as_dist: bool = True) -> tuple[float]:
    """
    Compute the overall stats of distances per layer
        [B, M, C, H, W] -> pair of [1, 1]
    :return:
    """
    from uutils.torch_uu import tensorify
    # - [B, L, 1], get distances per data sets (tasks) per layer
    distances_per_data_sets_per_layer: list[OrderedDict[LayerIdentifier, float]] = dist_batch_data_sets_for_all_layer(
        mdl1, mdl2, X1, X2,
        layer_names1,
        layer_names2, dist_type,
        downsample_method,
        downsample_size,
        effective_neuron_type,
        force_cpu, iters,
        metrics_as_dist)
    # list(OrderDict([B, L, 1])) -> list([B, L, 1])
    _distances_per_data_sets_per_layer: list[list[float]] = _dists_per_task_per_layer_to_list(
        distances_per_data_sets_per_layer)
    _distances_per_data_sets_per_layer: Tensor = tensorify(_distances_per_data_sets_per_layer)
    # - [B, L, 1] -> [1], get the avg distance/sim for each layer (and std)
    mu: Tensor = _distances_per_data_sets_per_layer.mean(dim=[0, 1])
    std: Tensor = _distances_per_data_sets_per_layer.std(dim=[0, 1])
    return mu, std


def pprint_results(mus: OrderedDict, stds: OrderedDict):
    print('---- stats of results per layer')
    print('-- mus (means) per layer')
    pprint(mus)
    print('-- stds (standard devs) per layer')
    pprint(stds)
    print()


# - helpers

def stats_compare_two_nets_per_layer(mdl1: nn.Module, mdl2: nn.Module,
                     X: Tensor,
                     layer_names: list[str],
                     dist_type: str,
                     downsample_method: Optional[str] = 'avg_pool',
                     downsample_size: Optional[int] = None,
                     effective_neuron_type: str = 'filter',
                     force_cpu: bool = False,
                     iters: int = 1,
                     metrics_as_dist: bool = True) -> tuple[OrderedDict]:
    mus, stds = stats_distance_per_layer(mdl1, mdl2, X, layer_names, layer_names, dist_type,
                                         downsample_method, downsample_size, effective_neuron_type,
                                         force_cpu, iters, metrics_as_dist)
    return mus, stds


def stats_compare_two_nets_entire_net(mdl1: nn.Module, mdl2: nn.Module,
                     X: Tensor,
                     layer_names: list[str],
                     dist_type: str,
                     downsample_method: Optional[str] = 'avg_pool',
                     downsample_size: Optional[int] = None,
                     effective_neuron_type: str = 'filter',
                     force_cpu: bool = False,
                     iters: int = 1,
                     metrics_as_dist: bool = True) -> tuple[float]:
    mus, stds = stats_distance_entire_net(mdl1, mdl2, X, layer_names, layer_names, dist_type,
                                         downsample_method, downsample_size, effective_neuron_type,
                                         force_cpu, iters, metrics_as_dist)
    return mus, stds

# - tests

def dist_per_layer_test():
    """
    """
    from copy import deepcopy

    import torch
    import torch.nn as nn

    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_from_default_args, \
        get_feature_extractor_layers
    from uutils.torch_uu import approx_equal

    # - very simple sanity check
    Cin: int = 3
    # num_out_filters: int = 8
    mdl1: nn.Module = get_default_learner_from_default_args()
    mdl2: nn.Module = deepcopy(mdl1)
    layer_names = get_feature_extractor_layers()
    print(f'{layer_names=}')

    # - ends up comparing two matrices of size [B, Dout], on same data, on same model
    B: int = 2
    M: int = 5  # e.g. k*n_c
    C, H, W = Cin, 84, 84
    downsample_size = None
    dist_type = 'pwcca'
    # -- same model same data, should be dist ~ 0.0
    X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, M, C, H, W))
    mus, stds = stats_distance_per_layer(mdl1, mdl1, X, X, layer_names, layer_names, dist_type,
                                         downsample_size=downsample_size)
    pprint_results(mus, stds)
    assert (mus != stds)
    mu, std = stats_distance_entire_net(mdl1, mdl1, X, X, layer_names, layer_names, dist_type,
                                        downsample_size=downsample_size)
    print(f'----entire net result: {mu=}, {std=}')
    assert (approx_equal(mu, 0.0))

    # -- differnt data different nets dist ~ 1.0, large distance
    X1: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, M, C, H, W))
    X2: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, M, C, H, W))
    mus, stds = stats_distance_per_layer(mdl1, mdl2, X1, X2, layer_names, layer_names, dist_type,
                                         downsample_size=downsample_size)
    pprint_results(mus, stds)
    assert (mus != stds)
    mu, std = stats_distance_entire_net(mdl1, mdl2, X1, X2, layer_names, layer_names, dist_type,
                                        downsample_size=downsample_size)
    print(f'----entire net result: {mu=}, {std=}')
    assert (approx_equal(mu, 1.0, tolerance=0.4))

    # -- same data same net, should be identical sim ~ 1.0
    metrics_as_dist: bool = False
    print(f'{metrics_as_dist=}')
    X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, M, C, H, W))
    mus, stds = stats_distance_per_layer(mdl1, mdl1, X, X, layer_names, layer_names, dist_type,
                                         downsample_size=downsample_size)
    pprint_results(mus, stds)
    assert (mus != stds)
    mu, std = stats_distance_entire_net(mdl1, mdl1, X, X, layer_names, layer_names, dist_type,
                                        downsample_size=downsample_size, metrics_as_dist=metrics_as_dist)
    print(f'----entire net result: {mu=}, {std=}')
    assert (approx_equal(mu, 1.0))


if __name__ == '__main__':
    import uutils

    start = time.time()
    dist_per_layer_test()
    print(f'time_passed_msg = {uutils.report_times(start)}')
    print('Done, success\a')
