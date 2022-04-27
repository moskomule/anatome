# import torch
# import numpy as np
# import random
#
# np.random.seed(0)
# torch.manual_seed(0)
# random.seed(0)

#%%
from copy import deepcopy

import torch
import torch.nn as nn

# import uutils.torch_uu
from uutils.torch_uu import get_metric, approx_equal
from uutils.torch_uu.models import get_named_identity_one_layer_linear_model

print('--- Sanity check: dCCA == 0.0 when using same reference to the same net with the same input. --')

Din: int = 10
Dout: int = Din
B: int = 2000
mdl1: nn.Module = get_named_identity_one_layer_linear_model(D=Din)
mdl2: nn.Module = mdl1
layer_name = 'fc0'

# - ends up comparing two matrices of size [B, Dout], on same data, on same model
metric_as_sim_or_dist: str = 'dist'
metric_comparison_type = 'svcca'
X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
dist: float = get_metric(mdl1, mdl2, X, X, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it close to 0.0? {approx_equal(dist, 0.0)}')
assert(approx_equal(dist, 0.0)), f'dist should be close to 0.0 but got {dist=}'

metric_comparison_type = 'pwcca'
X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
dist: float = get_metric(mdl1, mdl2, X, X, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it close to 0.0? {approx_equal(dist, 0.0)}')
assert(approx_equal(dist, 0.0)), f'dist should be close to 0.0 but got {dist=}'

metric_comparison_type = 'lincka'
X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
dist: float = get_metric(mdl1, mdl2, X, X, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it close to 0.0? {approx_equal(dist, 0.0)}')
assert(approx_equal(dist, 0.0)), f'dist should be close to 0.0 but got {dist=}'

metric_comparison_type = 'opd'
X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
dist: float = get_metric(mdl1, mdl2, X, X, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it close to 0.? {approx_equal(dist, 0.0, tolerance=1e-2)}')
assert(approx_equal(dist, 0.0, tolerance=1e-2)), f'dist should be close to 0.0 but got {dist=}'

#%%
from copy import deepcopy

import torch
import torch.nn as nn

# import uutils.torch_uu
from uutils.torch_uu import get_metric, approx_equal
from uutils.torch_uu.models import get_named_identity_one_layer_linear_model

print('--- Sanity check: dCCA == 0.0 when using the same net twice but different references same input (deepcopy) --')

Din: int = 10
Dout: int = Din
B: int = 2000
mdl1: nn.Module = get_named_identity_one_layer_linear_model(D=Din)
mdl2: nn.Module = deepcopy(mdl1)
layer_name = 'fc0'

# - ends up comparing two matrices of size [B, Dout], on same data, on same model
metric_as_sim_or_dist: str = 'dist'
metric_comparison_type = 'svcca'
X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
dist: float = get_metric(mdl1, mdl2, X, X, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it close to 0.0? {approx_equal(dist, 0.0)}')
assert(approx_equal(dist, 0.0)), f'dist should be close to 0.0 but got {dist=}'

metric_comparison_type = 'pwcca'
X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
dist: float = get_metric(mdl1, mdl2, X, X, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it close to 0.0? {approx_equal(dist, 0.0)}')
assert(approx_equal(dist, 0.0)), f'dist should be close to 0.0 but got {dist=}'

metric_comparison_type = 'lincka'
X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
dist: float = get_metric(mdl1, mdl2, X, X, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it close to 0.0? {approx_equal(dist, 0.0)}')
assert(approx_equal(dist, 0.0)), f'dist should be close to 0.0 but got {dist=}'

metric_comparison_type = 'opd'
X: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
dist: float = get_metric(mdl1, mdl2, X, X, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it close to 0.? {approx_equal(dist, 0.0, tolerance=1e-2)}')
assert(approx_equal(dist, 0.0, tolerance=1e-2)), f'dist should be close to 0.0 but got {dist=}'

#%%
from copy import deepcopy

import torch
import torch.nn as nn

# import uutils.torch_uu
from uutils.torch_uu import get_metric, approx_equal
from uutils.torch_uu.models import get_named_identity_one_layer_linear_model

print("--- Sanity check: dCCA == 0.0 when using same reference to the same network even though its different input ('BUG' CASE). --")

Din: int = 10
Dout: int = Din
B: int = 2000
mdl1: nn.Module = get_named_identity_one_layer_linear_model(D=Din)
mdl2: nn.Module = mdl1
layer_name = 'fc0'

# - ends up comparing two matrices of size [B, Dout], on same data, on same model
metric_as_sim_or_dist: str = 'dist'
metric_comparison_type = 'svcca'
X1: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
X2: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
dist: float = get_metric(mdl1, mdl2, X1, X2, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it close to 0.0? {approx_equal(dist, 0.0)}')
assert(approx_equal(dist, 0.0)), f'dist should be close to 0.0 but got {dist=}'

metric_comparison_type = 'pwcca'
dist: float = get_metric(mdl1, mdl2, X1, X2, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it close to 0.0? {approx_equal(dist, 0.0)}')
assert(approx_equal(dist, 0.0)), f'dist should be close to 0.0 but got {dist=}'

metric_comparison_type = 'lincka'
dist: float = get_metric(mdl1, mdl2, X1, X2, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it close to 0.0? {approx_equal(dist, 0.0)}')
assert(approx_equal(dist, 0.0)), f'dist should be close to 0.0 but got {dist=}'

metric_comparison_type = 'opd'
dist: float = get_metric(mdl1, mdl2, X1, X2, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'Should be very very close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it close to 0.? {approx_equal(dist, 0.0, tolerance=1e-2)}')
assert(approx_equal(dist, 0.0, tolerance=1e-2)), f'dist should be close to 0.0 but got {dist=}'

#%%
from copy import deepcopy

import torch
import torch.nn as nn

# import uutils.torch_uu as torch_uu
from uutils.torch_uu import norm
from uutils.torch_uu import get_metric, approx_equal
from uutils.torch_uu.models import get_named_identity_one_layer_linear_model

print("--- Sanity check: dCCA > 0.0 when using different reference to the same network and using different inputs. --")

Din: int = 10
Dout: int = Din
B: int = 2000
mdl1: nn.Module = get_named_identity_one_layer_linear_model(D=Din)
mdl2: nn.Module = deepcopy(mdl1)
layer_name = 'fc0'

# - ends up comparing two matrices of size [B, Dout], on same data, on same model
metric_as_sim_or_dist: str = 'dist'
X1: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
X2: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))

metric_comparison_type = 'svcca'
assert (X1.norm() != X2.norm())
assert norm(mdl1) == norm(mdl2), f'Models are same so they should have the same norm for weights bug got: {norm(mdl1),norm(mdl2)}'
dist: float = get_metric(mdl1, mdl2, X1, X2, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'{metric_as_sim_or_dist=}')
print(f'Should not be close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it far to 0.0? {not approx_equal(dist, 0.0)} it is: {dist=}')
assert(not approx_equal(dist, 0.0)), f' {dist=}'

metric_comparison_type = 'pwcca'
assert (X1.norm() != X2.norm())
assert norm(mdl1) == norm(mdl2), f'Models are same so they should have the same norm for weights bug got: {norm(mdl1),norm(mdl2)}'
dist: float = get_metric(mdl1, mdl2, X1, X2, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'{metric_as_sim_or_dist=}')
print(f'Should not be close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it far to 0.0? {not approx_equal(dist, 0.0)} it is: {dist=}')
assert(not approx_equal(dist, 0.0)), f' {dist=}'

metric_comparison_type = 'lincka'
assert (X1.norm() != X2.norm())
assert norm(mdl1) == norm(mdl2), f'Models are same so they should have the same norm for weights bug got: {norm(mdl1),norm(mdl2)}'
dist: float = get_metric(mdl1, mdl2, X1, X2, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'{metric_as_sim_or_dist=}')
print(f'Should not be close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it far to 0.0? {not approx_equal(dist, 0.0)} it is: {dist=}')
assert(not approx_equal(dist, 0.0)), f' {dist=}'

metric_comparison_type = 'opd'
assert (X1.norm() != X2.norm())
assert norm(mdl1) == norm(mdl2), f'Models are same so they should have the same norm for weights bug got: {norm(mdl1),norm(mdl2)}'
dist: float = get_metric(mdl1, mdl2, X1, X2, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'{metric_as_sim_or_dist=}')
print(f'Should not be close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it far to 0.0? {not approx_equal(dist, 0.0)} it is: {dist=}')
assert(not approx_equal(dist, 0.0)), f' {dist=}'

#%%
from copy import deepcopy

import torch
import torch.nn as nn

# import uutils.torch_uu as torch_uu
from uutils.torch_uu import norm
from uutils.torch_uu import get_metric, approx_equal
from uutils.torch_uu.models import get_named_identity_one_layer_linear_model

print("--- Sanity check: dCCA > 0.0 when using different reference to the same network and using different inputs. --")

Din: int = 10
Dout: int = Din
B: int = 2000
mdl1: nn.Module = get_named_identity_one_layer_linear_model(D=Din)
mdl2: nn.Module = deepcopy(mdl1)
# mdl2: nn.Module = mdl1
layer_name = 'fc0'

# - ends up comparing two matrices of size [B, Dout], on same data, on same model
metric_as_sim_or_dist: str = 'dist'
X1: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))
X2: torch.Tensor = torch.distributions.Normal(loc=0.0, scale=1.0).sample((B, Din))

metric_comparison_type = 'svcca'
assert (X1.norm() != X2.norm())
assert norm(mdl1) == norm(mdl2), f'Models are same so they should have the same norm for weights bug got: {norm(mdl1),norm(mdl2)}'
dist: float = get_metric(mdl1, mdl2, X1, X2, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'{metric_as_sim_or_dist=}')
print(f'Should not be close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it far to 0.0? {not approx_equal(dist, 0.0)} it is: {dist=}')
assert(not approx_equal(dist, 0.0)), f' {dist=}'

metric_comparison_type = 'pwcca'
assert (X1.norm() != X2.norm())
assert norm(mdl1) == norm(mdl2), f'Models are same so they should have the same norm for weights bug got: {norm(mdl1),norm(mdl2)}'
dist: float = get_metric(mdl1, mdl2, X1, X2, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'{metric_as_sim_or_dist=}')
print(f'Should not be close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it far to 0.0? {not approx_equal(dist, 0.0)} it is: {dist=}')
assert(not approx_equal(dist, 0.0)), f' {dist=}'

metric_comparison_type = 'lincka'
assert (X1.norm() != X2.norm())
assert norm(mdl1) == norm(mdl2), f'Models are same so they should have the same norm for weights bug got: {norm(mdl1),norm(mdl2)}'
dist: float = get_metric(mdl1, mdl2, X1, X2, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'{metric_as_sim_or_dist=}')
print(f'Should not be close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it far to 0.0? {not approx_equal(dist, 0.0)} it is: {dist=}')
assert(not approx_equal(dist, 0.0)), f' {dist=}'

metric_comparison_type = 'opd'
assert (X1.norm() != X2.norm())
assert norm(mdl1) == norm(mdl2), f'Models are same so they should have the same norm for weights bug got: {norm(mdl1),norm(mdl2)}'
dist: float = get_metric(mdl1, mdl2, X1, X2, layer_name, downsample_size=None, iters=1, metric_comparison_type=metric_comparison_type, metric_as_sim_or_dist=metric_as_sim_or_dist)
print(f'{metric_as_sim_or_dist=}')
print(f'Should not be close to 0.0: {dist=} ({metric_comparison_type=})')
print(f'Is it far to 0.0? {not approx_equal(dist, 0.0)} it is: {dist=}')
assert(not approx_equal(dist, 0.0)), f' {dist=}'