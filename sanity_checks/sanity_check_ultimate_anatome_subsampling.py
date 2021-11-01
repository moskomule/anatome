#%%
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

Cin: int = 3
num_out_filters: int = 8
mdl1: nn.Module = get_5cnn_model()
# mdl2: nn.Module = get_5cnn_model()
mdl2: nn.Module = deepcopy(mdl1)
layer_name = 'model.features.pool4'

metric_metric_comparison_type = 'svcca'
metric_comparison_type = 'pwcca'
effective_neuron_type: str = 'filter'

# -
B: int = 4
C, H, W = Cin, 84, 84
X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, C, H, W))

# - compute sim for NO downsample: so layer matrix is []
subsample_effective_num_data_method: str = 'subsampling_data_to_dims_ratio'
subsample_effective_num_data_param: int = 20
metric_as_sim_or_dist: str = 'sim'
sim: float = get_metric(mdl1, mdl2, X, X, layer_name,
                    metric_comparison_type=metric_comparison_type, effective_neuron_type=effective_neuron_type,
                     subsample_effective_num_data_method=subsample_effective_num_data_method,
                     subsample_effective_num_data_param=subsample_effective_num_data_param,
                     metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 1.0: {sim=} ({metric_comparison_type=})')
assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'

# # - compute sim for downsample
# sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=downsample_size, iters=1, cxa_dist_type=cxa_dist_type, effective_neuron_type=effective_neuron_type)
# print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
# assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'
#
# # - compute sim for downsample
# sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=downsample_size, iters=1, cxa_dist_type=cxa_dist_type, effective_neuron_type=effective_neuron_type)
# print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
# assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'
