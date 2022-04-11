"""
- helper functions to compute dists/sims for actual networks and experiments
"""
import math
from copy import deepcopy
from pprint import pprint

import torch
from typing import Union, Optional
from collections import OrderedDict

from torch import nn, Tensor


from pdb import set_trace as st

LayerIdentifier = Union[str, tuple[str]]


def compute_recommended_batch_size_for_trustworthy_experiments_for_neurons_as_activations(mdl: nn.Module) -> int:
    """

    Calculation:
        Make sure the standard N' >= safety * D' is true. For activations as neurons we get
            [M, C, H, W] -> [M, CHW]
        so we need to choose a downsample size such that N' >= s * D' is satisfied. For that we need:
        M >= s*CHW
        H=W=size
        M >= s*C*size^2
        where usually C, s are decided (e.g. C=32, S=10) and you need to choose a size such that M has
        enough data examples and the code runs (no OMM issues), the errors/stds are low enough etc.
        Plug in different size^2 code while the code computing the sims for entire (perhaps with meta-batches)
        until the OOM issue arises.

        Note: The only way to fix OOM is to fix the hook code or see if torch.fx fixes it.
    :param mdl:
    :return:
    """
    pass


def compute_recommended_batch_size_for_trustworthy_experiments_for_model_over_all_layers_for_neurons_as_filters(
        mdl: nn.Module) -> int:
    """
    Using torch.fx, loop through all the layers and computing the largest B recommnded. Most likely the H*W that is
    smallest woll win but formally just compute B_l for each layer that your computing sims/dists and then choose
    the largest B_l. That ceil(B_l) satisfies B*H*W >= s*C for all l since it's the largest.

    :param mdl:
    :return:
    """
    pass


def compute_recommended_batch_size_for_trustworthy_experiments(C: int, H: int, W: int, safety_val: float) -> int:
    """
    Based on inequality with safety_val=s:
        N' >= s*D'
    the recommended batch size is, assuming N'=B*H*W and D'=C (so considering neurons as filter, patches as data):
        B*H*W >= s*C
    leading to any batch size B that satisfies:
        B >= (s*C)/(H*W)
    for the current layer and model. So, C, H, W are for the current model at that layer.

    note:
        - recommended way to compute this is to get the largest B after plugging in the C, H, W for all the layers of
        your model - essentially computing the "worst-case" B needed for the model.
    :return:
    """
    recommended_batch_size: int = int(math.ceil(safety_val * C / (H * W)))
    assert (recommended_batch_size > 0), 'Batch size that was recommnded was negative, check the input your using.'
    return recommended_batch_size


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
    list([L]) -> OrderedDict([L])
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
    OrderedDict([L]) -> list([L])
    """
    return [dist for _, dist in dists_per_layer.items()]


def _dists_per_task_per_layer_to_list(dists_per_tasks_per_layer: list[OrderedDict[LayerIdentifier, float]]) -> list[
    list[float]]:
    """
    OrderedDict([B, L]) -> list([B, L])
    """
    B: int = len(dists_per_tasks_per_layer)
    _dists_per_tasks_per_layer: list[list[float]] = []
    for b in range(B):
        _dists_per_layer: list[float] = _dists_per_layer_to_list(dists_per_tasks_per_layer[b])
        _dists_per_tasks_per_layer.append(_dists_per_layer)
    return _dists_per_tasks_per_layer


# def _dists_to_tensor():
#     distances_per_data_sets_per_layer: list[list[float]] = _dists_per_task_per_layer_to_list(
#         distances_per_data_sets_per_layer)
#     distances_per_data_sets_per_layer: Tensor = tensorify(distances_per_data_sets_per_layer)

def dist_data_set_per_layer(mdl1: nn.Module, mdl2: nn.Module,
                            X1: Tensor, X2: Tensor,
                            layer_names1: list[str], layer_names2: list[str],
                            metric_comparison_type: str = 'pwcca',
                            iters: int = 1,
                            effective_neuron_type: str = 'filter',
                            downsample_method: Optional[str] = None,
                            downsample_size: Optional[int] = None,
                            subsample_effective_num_data_method: Optional[str] = None,
                            subsample_effective_num_data_param: Optional[int] = None,
                            metric_as_sim_or_dist: str = 'dist',
                            force_cpu: bool = False
                            ) -> OrderedDict[LayerIdentifier, float]:
    """
    Given a pair of data sets (or batches, potentially the same one), compute the distance btw the models
        [M, C, H, W], [L] -> [L] (as list like obj)
    assuming the list of layes are in the order of how you want to compute the comparison.
    E.g.
        - d(f_ml(X), A(f_ml, task)(X)), compute the distance between model and adapted model for same set of examples
        for the task/data set since X1=X2=X.
        - dv = d(f(X1), f(X2))
        - dv = d(f^*_1(X1), f^*_2(X2))

    :return: [L, 1] as a iterable (e.g. list of floats per layer) correponding to the distance for each layer.
    """
    assert metric_as_sim_or_dist == 'dist'
    from anatome.similarity import SimilarityHook
    assert len(layer_names1) == len(layer_names2), f'Expected same number of layers for comparing both nets but got: ' \
                                                   f'{(len(layer_names1))=}, {len(layer_names2)=}'
    # - create hooks for each layer name
    # mdl1 = mdl1
    # mdl2 = deepcopy(mdl2)
    print(f'{(mdl1 is mdl2)=}')
    hooks1: list[SimilarityHook] = SimilarityHook.create_hooks(mdl1, layer_names1, metric_comparison_type, force_cpu)
    hooks2: list[SimilarityHook] = SimilarityHook.create_hooks(mdl2, layer_names2, metric_comparison_type, force_cpu)
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
        dist: float = hook1.distance(hook2,
                                     effective_neuron_type=effective_neuron_type,
                                     downsample_method=downsample_method,
                                     downsample_size=downsample_size,
                                     subsample_effective_num_data_method=subsample_effective_num_data_method,
                                     subsample_effective_num_data_param=subsample_effective_num_data_param,
                                     metric_as_sim_or_dist=metric_as_sim_or_dist
                                     )
        layer_key: LayerIdentifier = layer1 if layer1 == layer2 else (layer1, layer2)
        dists_od[layer_key] = dist
    assert len(dists_od) == len(layer_names1)
    _clear_hooks(hooks1)
    _clear_hooks(hooks2)
    return dists_od


def dist_batch_data_sets_for_all_layer(mdl1: nn.Module, mdl2: nn.Module,
                                       X1: Tensor, X2: Tensor,
                                       layer_names1: list[str], layer_names2: list[str],
                                       metric_comparison_type: str = 'pwcca',
                                       iters: int = 1,
                                       effective_neuron_type: str = 'filter',
                                       downsample_method: Optional[str] = None,
                                       downsample_size: Optional[int] = None,
                                       subsample_effective_num_data_method: Optional[str] = None,
                                       subsample_effective_num_data_param: Optional[int] = None,
                                       metric_as_sim_or_dist: str = 'dist',
                                       force_cpu: bool = False
                                       ) -> list[OrderedDict[LayerIdentifier, float]]:
    """
    Gets the distance for a batch (e.g meta-batch) of data sets/tasks and for each layer the distances between the nets:
        [B, M, C, H, W], [L] -> list(OrderDict([B, L]))
    Example use:
        - Computing d(f_ml(X), A(f_ml, t)(X)) so X1=X2=X.
        - Computing dv so some restricted cross product of the data sets in X1 and X2 (X1 != X2),
        ideally avoiding diagonal if dv.

    :return:
    """
    assert metric_as_sim_or_dist == 'dist'
    assert len(X1.size()) == 5, f'Data set does not 5 dims i.e. [B, M, C, H, W] instead got: {X1.size()=}'
    assert len(X2.size()) == 5, f'Data set does not 5 dims i.e. [B, M, C, H, W] instead got: {X2.size()=}'
    # - get distances per data sets/tasks per layers: [B, M, C, H, W], [L] -> [B, L]
    B: int = min(X1.size(0), X2.size(0))
    dists_entire_net_all_data_sets: list[OrderedDict[LayerIdentifier, float]] = []  # [B, L]
    for b in range(B):
        # - [B, M, C, H, W] -> [M, C, H, W] (get a specific data set/batch)
        x1, x2 = X1[b], X2[b]
        # - compute for current pair of data sets (or tasks) the distance between models
        # [M, C, H, W], [L] -> [L], really a list of floats of len [L]
        dist_for_data_set_b: OrderedDict[LayerIdentifier, float] = dist_data_set_per_layer(mdl1, mdl2, x1, x2,
                                                                                           layer_names1, layer_names2,
                                                                                           metric_comparison_type=metric_comparison_type,
                                                                                           iters=iters,
                                                                                           effective_neuron_type=effective_neuron_type,
                                                                                           downsample_method=downsample_method,
                                                                                           downsample_size=downsample_size,
                                                                                           subsample_effective_num_data_method=subsample_effective_num_data_method,
                                                                                           subsample_effective_num_data_param=subsample_effective_num_data_param,
                                                                                           metric_as_sim_or_dist=metric_as_sim_or_dist,
                                                                                           force_cpu=force_cpu
                                                                                           )
        # adding to [B, L]
        dists_entire_net_all_data_sets.append(dist_for_data_set_b)
    # check effective size [B, L]
    L: int = len(layer_names1)
    assert len(dists_entire_net_all_data_sets) == B
    assert len(dists_entire_net_all_data_sets[0]) == L
    # - [B, L], return list of iters of distances per data sets (tasks) per layers
    return dists_entire_net_all_data_sets


def compute_stats_from_distance_per_batch_of_data_sets_per_layer(
        distances_per_data_sets_per_layer: list[OrderedDict[LayerIdentifier, float]],
        dist2sim: bool = False) -> tuple[OrderedDict[LayerIdentifier, float], OrderedDict[LayerIdentifier, float]]:
    """
    Given a list of distances [B, L] compute the diversity (per layer) [L].

    [B, L] -> [L]^2, means and cis
    """
    from uutils.torch_uu.metrics.confidence_intervals import torch_compute_confidence_interval
    from uutils.torch_uu import tensorify
    # - get layer ids
    layer_ids: list[LayerIdentifier] = list(distances_per_data_sets_per_layer[0].keys())
    layer_names1 = layer_ids
    L: int = len(layer_names1)
    B: int = len(distances_per_data_sets_per_layer)

    # -list(OrderDict([B, L])) -> list([B, L])
    distances_per_data_sets_per_layer: list[list[float]] = _dists_per_task_per_layer_to_list(
        distances_per_data_sets_per_layer)
    distances_per_data_sets_per_layer: Tensor = tensorify(distances_per_data_sets_per_layer)
    # if dist2sim:
    #     # dist -> sim
    #     distances_per_data_sets_per_layer: Tensor = 1.0 - distances_per_data_sets_per_layer
    assert (distances_per_data_sets_per_layer.dim() == 2)
    assert len(distances_per_data_sets_per_layer) == B
    assert len(distances_per_data_sets_per_layer[0]) == L

    # - [B, L] -> [L] compute expectation of distance tasks (i.e. div) & cis for each layer
    mu_ci_per_layer: list[tuple[Tensor, Tensor]] = [torch_compute_confidence_interval(distances_per_data_sets_per_layer[:, l]) for l in range(L)]
    means_per_layer = tensorify([mu for mu, _ in mu_ci_per_layer])
    cis_per_layer = tensorify([ci for _, ci in mu_ci_per_layer])
    assert means_per_layer.size() == torch.Size([L])
    assert cis_per_layer.size() == torch.Size([L])

    # [L, 1] -> OrderDict([L])
    assert len(layer_ids) == L == len(layer_names1)
    mean_distance_per_layer: OrderedDict[LayerIdentifier, float] = OrderedDict()
    ci_distance_per_layer: OrderedDict[LayerIdentifier, float] = OrderedDict()
    for l_idx in range(L):
        layer_id: LayerIdentifier = layer_ids[l_idx]
        mu: float = means_per_layer[l_idx].item()
        ci: float = cis_per_layer[l_idx].item()
        mean_distance_per_layer[layer_id] = mu
        ci_distance_per_layer[layer_id] = ci
    assert len(mean_distance_per_layer) == L
    assert len(ci_distance_per_layer) == L
    return mean_distance_per_layer, ci_distance_per_layer


def compute_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks(
        distances_per_data_sets_per_layer: list[OrderedDict[LayerIdentifier, float]], dist2sim: bool = False) \
        -> tuple[float, float]:
    """
        [B, L] -> mu, std
    where mu = 1/B*L sum_{b,l} distances_per_data_sets_per_layer[b, l]
    where std = 1/B*L sum_{b,l} (distances_per_data_sets_per_layer[b, l]-mu)^2
    :param distances_per_data_sets_per_layer:
    :return:
    """
    from uutils.torch_uu import tensorify

    # -list(OrderDict([B, L])) -> list([B, L])
    distances_per_data_sets_per_layer: list[list[float]] = _dists_per_task_per_layer_to_list(
        distances_per_data_sets_per_layer)
    distances_per_data_sets_per_layer: Tensor = tensorify(distances_per_data_sets_per_layer)
    if dist2sim:
        # dist -> sim
        distances_per_data_sets_per_layer: Tensor = 1.0 - distances_per_data_sets_per_layer
    assert (distances_per_data_sets_per_layer.dim() == 2)

    # - [B, L, 1] -> [1], get the avg distance/sim for each layer (and std)
    mu: Tensor = distances_per_data_sets_per_layer.mean(dim=[0, 1])
    std: Tensor = distances_per_data_sets_per_layer.std(dim=[0, 1])
    assert (mu.size() == torch.Size([]))
    assert (std.size() == torch.Size([]))
    return mu, std


# def metrics2opposite_metrics(metrics_per_data_sets_per_layer: list[OrderedDict[LayerIdentifier, float]])\
#         -> list[OrderedDict[LayerIdentifier, float]]:
#     """
#     Converts sim2dists and dists2sim by doing metric - 1.0 and returning same list[Ordered] object.
#     :param metrics_per_data_sets_per_layer:
#     :return:
#     """
#     #
#     new_metrics_per_data_sets_per_layer: list[OrderedDict[LayerIdentifier, float]] = deepcopy(metrics_per_data_sets_per_layer)
#     for i, order_dict_layer2float in enumerate(metrics_per_data_sets_per_layer):
#         assert len(new_metrics_per_data_sets_per_layer[i]) == len(metrics_per_data_sets_per_layer[i])
#         for layer_name, metric in order_dict_layer2float.items():
#             new_metrics_per_data_sets_per_layer[i][layer_name] = 1.0 - metric
#             assert len(new_metrics_per_data_sets_per_layer[i]) == len(metrics_per_data_sets_per_layer[i])
#     assert len(new_metrics_per_data_sets_per_layer) == len(metrics_per_data_sets_per_layer)
#     return new_metrics_per_data_sets_per_layer


def pprint_results(mus: OrderedDict, stds: OrderedDict):
    print('---- stats of results per layer')
    print('-- mus (means) per layer')
    pprint(mus)
    print('-- stds (standard devs) per layer')
    pprint(stds)
    print()

# - visualization helper
# --> See ultimate-utils plot __init__ file.

# - tests

# def dist_per_layer_test():
#     """
#     """
#     from copy import deepcopy
#
#     import torch
#     import torch.nn as nn
#
#     from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_from_default_args, \
#         get_feature_extractor_conv_layers
#     from uutils.torch_uu import approx_equal
#
#     # - very simple sanity check
#     Cin: int = 3
#     # num_out_filters: int = 8
#     mdl1: nn.Module = get_default_learner_from_default_args()
#     mdl2: nn.Module = deepcopy(mdl1)
#     layer_names = get_feature_extractor_conv_layers()
#     print(f'{layer_names=}')
#
#     # - ends up comparing two matrices of size [B, Dout], on same data, on same model
#     B: int = 5  # -- satisfies B >= (10*32)/(5**2) = 12.8 for this specific 5CNN model
#     M: int = 13  # e.g. k*n_c, satisfies B >= (10*32)/(5**2) = 12.8 for this specific 5CNN model
#     C, H, W = Cin, 84, 84
#     metric_comparison_type: str = 'pwcca'
#     X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, M, C, H, W))
#     effective_neuron_type: str = 'filter'
#     subsample_effective_num_data_method: str = 'subsampling_data_to_dims_ratio'
#     subsample_effective_num_data_param: int = 10
#     metric_as_sim_or_dist: str = 'sim'
#     mus, stds, distances_per_data_sets_per_layer = stats_distance_per_layer(mdl1, mdl1, X, X, layer_names, layer_names,
#                                                                             metric_comparison_type=metric_comparison_type,
#                                                                             effective_neuron_type=effective_neuron_type,
#                                                                             subsample_effective_num_data_method=subsample_effective_num_data_method,
#                                                                             subsample_effective_num_data_param=subsample_effective_num_data_param,
#                                                                             metric_as_sim_or_dist=metric_as_sim_or_dist)
#     pprint_results(mus, stds)
#     assert (mus != stds)
#     mu, std = _stats_distance_entire_net(mdl1, mdl1, X, X, layer_names, layer_names,
#                                          metric_comparison_type=metric_comparison_type,
#                                          effective_neuron_type=effective_neuron_type,
#                                          subsample_effective_num_data_method=subsample_effective_num_data_method,
#                                          subsample_effective_num_data_param=subsample_effective_num_data_param,
#                                          metric_as_sim_or_dist=metric_as_sim_or_dist)
#     print(f'----entire net result: {mu=}, {std=}')
#     assert (approx_equal(mu, 1.0))
#     mu2, std2 = compute_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks(distances_per_data_sets_per_layer)
#     assert (approx_equal(mu, mu2))
#     assert (approx_equal(std, std2))
#
#     # -- differnt data different nets dist ~ 1.0, large distance, so low sim
#     X1: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, M, C, H, W))
#     X2: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, M, C, H, W))
#     mus, stds, distances_per_data_sets_per_layer = stats_distance_per_layer(mdl1, mdl2, X1, X2, layer_names,
#                                                                             layer_names,
#                                                                             metric_comparison_type=metric_comparison_type,
#                                                                             effective_neuron_type=effective_neuron_type,
#                                                                             subsample_effective_num_data_method=subsample_effective_num_data_method,
#                                                                             subsample_effective_num_data_param=subsample_effective_num_data_param,
#                                                                             metric_as_sim_or_dist=metric_as_sim_or_dist)
#     pprint_results(mus, stds)
#     assert (mus != stds)
#     mu, std = _stats_distance_entire_net(mdl1, mdl2, X1, X2, layer_names, layer_names,
#                                          metric_comparison_type=metric_comparison_type,
#                                          effective_neuron_type=effective_neuron_type,
#                                          subsample_effective_num_data_method=subsample_effective_num_data_method,
#                                          subsample_effective_num_data_param=subsample_effective_num_data_param,
#                                          metric_as_sim_or_dist=metric_as_sim_or_dist)
#     print(f'----entire net result: {mu=}, {std=}')
#     assert (approx_equal(mu, 0.0, tolerance=0.4))
#     mu2, std2 = compute_mu_std_for_entire_net_from_all_distances_from_data_sets_tasks(distances_per_data_sets_per_layer)
#     assert (approx_equal(mu, mu2))
#     assert (approx_equal(std, std2))


if __name__ == '__main__':
    import uutils
    import time

    start = time.time()
    # dist_per_layer_test()
    print(f'time_passed_msg = {uutils.report_times(start)}')
    print('Done, success\a')
