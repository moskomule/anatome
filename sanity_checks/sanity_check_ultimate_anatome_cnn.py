#%%

"""
The similarity of the same network should always be 1.0 on same input.
"""
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

from uutils.torch_uu import cxa_sim, approx_equal
from uutils.torch_uu.models import get_single_conv_model
import anatome
print(f'from import: {anatome=}')

print('--- Sanity check: sCCA = 1.0 when using same net twice with same input. --')

Cin: int = 3
num_out_filters: int = 8
conv_layer: nn.Module = get_single_conv_model(in_channels=3, num_out_filters=num_out_filters)
mdl1: nn.Module = nn.Sequential(OrderedDict([('conv1', conv_layer)]))
mdl2: nn.Module = deepcopy(mdl1)
layer_name = 'conv1'

# - ends up comparing two matrices of size [B, Dout], on same data, on same model
B: int = 4
C, H, W = Cin, 64, 64
# downsample_size = None
cxa_dist_type = 'svcca'
cxa_dist_type = 'pwcca'
X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, C, H, W))

# - compute sim for NO downsample: so layer matrix is []
downsample_size = None
sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=downsample_size, iters=1, cxa_dist_type=cxa_dist_type)
print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'

# - compute sim for downsample
downsample_size: int = 5
sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=downsample_size, iters=1, cxa_dist_type=cxa_dist_type)
print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'

# - compute sim for downsample
downsample_size: int = 1
sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=downsample_size, iters=1, cxa_dist_type=cxa_dist_type)
print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'


#%%
"""
ultimate anatome with activation as sim
"""
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

from uutils.torch_uu import cxa_sim, approx_equal
from uutils.torch_uu.models import get_single_conv_model
import anatome

effective_neuron_type: str = 'activation'

# - compute sim for NO downsample: so layer matrix is []
downsample_size = None
sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=downsample_size, iters=1, cxa_dist_type=cxa_dist_type, effective_neuron_type=effective_neuron_type)
print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'

# - compute sim for downsample
downsample_size: int = 5
sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=downsample_size, iters=1, cxa_dist_type=cxa_dist_type, effective_neuron_type=effective_neuron_type)
print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'

# - compute sim for downsample
downsample_size: int = 1
sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=downsample_size, iters=1, cxa_dist_type=cxa_dist_type, effective_neuron_type=effective_neuron_type)
print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'


#%%
"""
ultimate with original anatome as sim
"""
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn

from uutils.torch_uu import cxa_sim, approx_equal
from uutils.torch_uu.models import get_single_conv_model
import anatome

from uutils.torch_uu import cxa_sim, approx_equal
from uutils.torch_uu.models import get_single_conv_model
import anatome

effective_neuron_type: str = 'original_anatome'

# - compute sim for NO downsample: so layer matrix is []
downsample_size = None
sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=downsample_size, iters=1, cxa_dist_type=cxa_dist_type, effective_neuron_type=effective_neuron_type)
print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'

# - compute sim for downsample
downsample_size: int = 5
sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=downsample_size, iters=1, cxa_dist_type=cxa_dist_type, effective_neuron_type=effective_neuron_type)
print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'

# - compute sim for downsample
downsample_size: int = 1
sim: float = cxa_sim(mdl1, mdl2, X, layer_name, downsample_size=downsample_size, iters=1, cxa_dist_type=cxa_dist_type, effective_neuron_type=effective_neuron_type)
print(f'Should be very very close to 1.0: {sim=} ({cxa_dist_type=})')
assert(approx_equal(sim, 1.0)), f'Sim should be close to 1.0 but got {sim=}'