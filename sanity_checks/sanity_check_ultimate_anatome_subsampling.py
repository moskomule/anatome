# %%
"""
subsample the effective data dimension.
"""
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

from uutils.torch_uu import approx_equal, get_metric
from uutils.torch_uu.models import get_5cnn_model
import anatome

mdl1: nn.Module = get_5cnn_model()
# mdl2: nn.Module = get_5cnn_model()
mdl2: nn.Module = deepcopy(mdl1)
# layer_name = 'model.features.conv1'
# layer_name = 'model.features.relu4'
layer_name = 'model.features.pool4'

metric_metric_comparison_type = 'svcca'
metric_comparison_type = 'pwcca'

# -
B: int = 13  # -- satisfies B >= (10*32)/(5**2) = 12.8 for this specific 5CNN model
C, H, W = 3, 84, 84
X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, C, H, W))

# - compute sims
effective_neuron_type: str = 'filter'
subsample_effective_num_data_method: str = 'subsampling_data_to_dims_ratio'
subsample_effective_num_data_param: int = 10
metric_as_sim_or_dist: str = 'sim'
sim: float = get_metric(mdl1, mdl2, X, X, layer_name,
                        metric_comparison_type=metric_comparison_type, effective_neuron_type=effective_neuron_type,
                        subsample_effective_num_data_method=subsample_effective_num_data_method,
                        subsample_effective_num_data_param=subsample_effective_num_data_param,
                        metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 1.0: {sim=} ({metric_comparison_type=})')
assert (approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'

# - compute sims
effective_neuron_type: str = 'filter'
subsample_effective_num_data_method: str = 'subsampling_size'
subsample_effective_num_data_param: int = 13
metric_as_sim_or_dist: str = 'sim'
sim: float = get_metric(mdl1, mdl2, X, X, layer_name,
                        metric_comparison_type=metric_comparison_type, effective_neuron_type=effective_neuron_type,
                        subsample_effective_num_data_method=subsample_effective_num_data_method,
                        subsample_effective_num_data_param=subsample_effective_num_data_param,
                        metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 1.0: {sim=} ({metric_comparison_type=})')
assert (approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'

# - compute sims
effective_neuron_type: str = 'filter'
downsample_method: int = 'avg_pool'
downsample_size: int = 4
metric_as_sim_or_dist: str = 'sim'
sim: float = get_metric(mdl1, mdl2, X, X, layer_name,
                        metric_comparison_type=metric_comparison_type, effective_neuron_type=effective_neuron_type,
                        downsample_method=downsample_method,
                        downsample_size=downsample_size,
                        metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 1.0: {sim=} ({metric_comparison_type=})')
assert (approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'


# - compute sims
effective_neuron_type: str = 'activation'
downsample_method: int = 'avg_pool'
downsample_size: int = 4
metric_as_sim_or_dist: str = 'sim'
sim: float = get_metric(mdl1, mdl2, X, X, layer_name,
                        metric_comparison_type=metric_comparison_type, effective_neuron_type=effective_neuron_type,
                        downsample_method=downsample_method,
                        downsample_size=downsample_size,
                        metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 1.0: {sim=} ({metric_comparison_type=})')
assert (approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'

# - compute sims
effective_neuron_type: str = 'original_anatome'
downsample_method: int = 'avg_pool'
downsample_size: int = 4
metric_as_sim_or_dist: str = 'sim'
sim: float = get_metric(mdl1, mdl2, X, X, layer_name,
                        metric_comparison_type=metric_comparison_type, effective_neuron_type=effective_neuron_type,
                        downsample_method=downsample_method,
                        downsample_size=downsample_size,
                        metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 1.0: {sim=} ({metric_comparison_type=})')
assert (approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'

# - compute dists
effective_neuron_type: str = 'original_anatome'
downsample_method: int = 'avg_pool'
downsample_size: int = 4
metric_as_sim_or_dist: str = 'dist'
dist: float = get_metric(mdl1, mdl2, X, X, layer_name,
                        metric_comparison_type=metric_comparison_type, effective_neuron_type=effective_neuron_type,
                        downsample_method=downsample_method,
                        downsample_size=downsample_size,
                        metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 0.0: {dist=} ({metric_comparison_type=})')
assert (approx_equal(dist, 0.0)), f'Sim should be close to 0.0 but got {dist=}'

# - compute dists
effective_neuron_type: str = 'filter'
downsample_method: int = 'avg_pool'
downsample_size: int = 4
metric_as_sim_or_dist: str = 'dist'
dist: float = get_metric(mdl1, mdl2, X, X, layer_name,
                        metric_comparison_type=metric_comparison_type, effective_neuron_type=effective_neuron_type,
                        downsample_method=downsample_method,
                        downsample_size=downsample_size,
                        metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 0.0: {dist=} ({metric_comparison_type=})')
assert (approx_equal(dist, 0.0)), f'Sim should be close to 0.0 but got {dist=}'
