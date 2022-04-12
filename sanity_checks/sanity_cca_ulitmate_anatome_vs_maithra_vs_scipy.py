"""
Some experiment results seem funny with PWCCA so let's make sure the CCA values for of ultimate anatome matches or is
close to values of other code.

conda install pytorch torchvision torchaudio -c pytorch

conda install pytorch torchvision torchaudio -c pytorch==1.10

refs:
    - https://github.com/moskomule/anatome/issues/30
    - https://github.com/moskomule/anatome/issues/27
    - https://stackoverflow.com/questions/69993768/how-does-one-implement-pwcca-in-pytorch-match-the-original-pwcca-implemented-in
"""
#%%
import sys
from pprint import pprint

import torch
from matplotlib import pyplot as plt

from uutils.torch_uu import approx_equal
from uutils.torch_uu.metrics.cca import cca_core, pwcca

from torch import Tensor
from anatome.similarity import svcca_distance, _cca_by_svd, _cca_by_qr, _compute_cca_traditional_equation, cca, \
    svcca_distance_keeping_fixed_dims, pwcca_distance_choose_best_layer_matrix
# from anatome.distance import svcca_distance, cca_by_svd, cca_by_qr
import numpy as np
import random

from uutils.torch_uu.metrics.cca.uutils_cca_core_addendums import svcca_with_keeping_fixed_dims, center


from pdb import set_trace as st

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# - checking anatome being uses
print('-- anatome being used --')
import os
os.system('pip list | grep anatome')
import anatome
print(f'{anatome=}')
pprint(f'{sys.path=}')

print('\n-- start cca test --')

# b1 = np.random.randn(*acts1.shape)
# b2 = np.random.randn(*acts2.shape)
print('NOTE: original tutorial does NOT center random baseline for these experiments, probably a combination of'
      'carelessness/bug or that it doesnt matter for these since b1, b2 are already centered. '
      'Tough the also forgot to center the MNIST activations, but due to BN I suspect it doesnt make a big difference.')

N = 1_000
D1, D2 = 50, 60

# - creating a random baseline
b1 = np.random.randn(D1, N)
b2 = np.random.randn(D2, N)

# we center for consistency, anatome also centers again, but it shouldn't matter
b1 = center(b1)
print(b1)
b2 = center(b2)
b1_t = torch.from_numpy(b1)
b2_t = torch.from_numpy(b2)
print(b1_t)
b1_t = b1_t.T
b2_t = b2_t.T
assert(b1_t.size() == torch.Size([N, D1]))
print('-- reproducibity finger print')
print(f'{b1.sum()=}')
print(f'{b2.sum()=}')
print(f'{b1_t.sum()=}')
print(f'{b2_t.sum()=}')
print(f'{b1.shape=}')
print(f'{b1_t.shape=}')

dims_to_keep: int = min(5, max(D1, D2))

# ---- CCA test
# - get cca values for baseline
print("\n-- Google's CCA -- ")
baseline = cca_core.get_cca_similarity(b1, b2, epsilon=1e-10, verbose=False)
# _plot_helper(baseline["cca_coef1"], "CCA coef idx", "CCA coef value")
# print("Baseline Mean CCA similarity", np.mean(baseline["cca_coef1"]))
# print("Baseline CCA similarity", baseline["cca_coef1"])
print(f'{len(baseline["cca_coef1"])=}')
print("Baseline CCA similarity", baseline["cca_coef1"][:dims_to_keep])
print(f"{np.mean(baseline['cca_coef1'])=}")

# - get sklern's cca's https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html, https://stackoverflow.com/questions/69800500/how-to-calculate-correlation-coefficients-using-sklearn-cca-module
# from sklearn.cross_decomposition import CCA
# # cca = CCA(n_components=D)
# cca = CCA(n_components=6)
# cca.fit(b1, b2)

# -
print("\n-- Ultimate Anatome's CCA --")
a, b, diag = cca(x=b1_t, y=b2_t, backend='svd')
# a, b, diag = cca_by_svd(b1_t, b2_t)
# a, b, diag = cca_by_qr(b1_t, b2_t)
print(f'{diag.size()=}')
print(f'{diag[:dims_to_keep]=}')
print(f'{diag.mean()=}')
assert (approx_equal(diag.mean(), np.mean(baseline['cca_coef1'])))

# - extra test
svcca_uanatome: Tensor = 1.0 - svcca_distance(x=b1_t, y=b2_t, accept_rate=1.0, backend='svd')
print(f"extra test with acceptance_rate=1.0: {svcca_uanatome=} should be close to {np.mean(baseline['cca_coef1'])}")
# assert approx_equal(svcca_uanatome, np.mean(baseline['cca_coef1']), 0.05)

# -
# print("\n-- Tranditional CCA eqs in PyTorch --")
# diag = _compute_cca_traditional_equation(b1_t, b2_t)
# print(f'{diag.size()=}')
# print(f'{diag[:dims_to_keep]=}')
# print(f'{diag.mean()=}')


# ---- SVCCA test1 (# of num_dims to keep hardcoded instead of via variance to keep)
# -
# print("\n\n---- Google's SVCCA test1 (# of num_dims to keep hardcoded instead of via variance to keep) ---- ")
# svcca_baseline = svcca_with_keeping_fixed_dims(x=b1, y=b2, dims_to_keep=dims_to_keep)
# svcca_keeping_fixed_dims: float = np.mean(svcca_baseline["cca_coef1"])
# print(f'{svcca_keeping_fixed_dims=}')
#
# # -
# print("\n---- Ultimate Anatome's SVCCA test1 (# of num_dims to keep hardcoded instead of via variance to keep) ----")
# svcca_uanatome_keeping_fixed_dims_original_anatome: Tensor = 1.0 - svcca_distance_keeping_fixed_dims(x=b1_t, y=b2_t,
#                                                                                     num=dims_to_keep,
#                                                                                     backend='svd',
#                                                                                     reduce_backend='original_anatome')
# print(f'{svcca_uanatome_keeping_fixed_dims_original_anatome=}')
# svcca_uanatome_keeping_fixed_dims_original_svcca: Tensor = 1.0 - svcca_distance_keeping_fixed_dims(x=b1_t, y=b2_t,
#                                                                                     num=dims_to_keep,
#                                                                                     backend='svd',
#                                                                                     reduce_backend='original_svcca')
# print(f'{svcca_uanatome_keeping_fixed_dims_original_svcca=}')
# assert approx_equal(svcca_uanatome_keeping_fixed_dims_original_anatome, svcca_keeping_fixed_dims, 0.01)

# ---- SVCCA test2 (# keep .99 of variance)
# -
print("\n\n------ Google's SVCCA test2 (# keep .99 of variance) ------ ")
# ---- PWCCA test2 (# keep .99 of variance)
# -
print("\n------ Ultimate Anatome's SVCCA test2 (# keep .99 of variance) ------")

# ---- PWCCA test3
# -
print("\n\n------ Google's PWCCA test  ------ ")
pwcca_mean, w, _ = pwcca.compute_pwcca(b1, b2, epsilon=1e-10)
print(f'{pwcca_mean=}')
pwcca_mean2, w, _ = pwcca.compute_pwcca2(b1, b2, epsilon=1e-10)
print(f'{pwcca_mean2=}')

# ---- PWCCA test3
# -
print("\n------ Ultimate Anatome's PWCCA test ------")
# pwcca_ultimateanatome: Tensor = 1.0 - pwcca_distance(x=b1_t, y=b2_t, backend='svd')
# print(f'{pwcca_ultimateanatome=}')
# pwcca_ultimateanatome: Tensor = 1.0 - pwcca_distance2(b1_t, b2_t, backend='svd', use_layer_matrix='L1')
# print(f'L1: {pwcca_ultimateanatome=}')
# pwcca_ultimateanatome: Tensor = 1.0 - pwcca_distance2(b1_t, b2_t, backend='svd', use_layer_matrix='L2')
# print(f'L2: {pwcca_ultimateanatome=}')
# pwcca_ultimateanatome: Tensor = 1.0 - pwcca_distance3(b1_t, b2_t)
# print(f'{pwcca_ultimateanatome=}')

# --
print('\n-------- do they match? --------')
print(f'{b1.sum()=}')
print(f'{b2.sum()=}')
print(f'{b1_t.sum()=}')
print(f'{b2_t.sum()=}')
pwcca_mean, w, _ = pwcca.compute_pwcca(acts1=b1, acts2=b2, epsilon=1e-10)
pwcca_mean2, w, _ = pwcca.compute_pwcca2(acts1=b1, acts2=b2, epsilon=1e-10)
# pwcca_ultimateanatome: Tensor = 1.0 - pwcca_distance_choose_best_layer_matrix(L1=b1_t, L2=b2_t, backend='svd', epsilon=1e-10)
# pwcca_ultimateanatome: Tensor = 1.0 - pwcca_distance2(x=b1_t, y=b2_t, backend='svd')
pwcca_ultimateanatome: Tensor = 1.0 - pwcca_distance_choose_best_layer_matrix(x=b1_t, y=b2_t, backend='svd')
pwcca_ultimateanatome_L1: Tensor = 1.0 - pwcca_distance_choose_best_layer_matrix(x=b1_t, y=b2_t, backend='svd', use_layer_matrix='x')
pwcca_ultimateanatome_L2: Tensor = 1.0 - pwcca_distance_choose_best_layer_matrix(x=b1_t, y=b2_t, backend='svd', use_layer_matrix='y')
# pwcca_extended_anatome: Tensor = 1.0 - pwcca_distance_extended_original_anatome(x=b1_t, y=b2_t, backend='svd')
# pwcca_extended_anatome_L1: Tensor = 1.0 - pwcca_distance_extended_original_anatome(x=b1_t, y=b2_t, backend='svd', use_layer_matrix='x')
# pwcca_extended_anatome_L2: Tensor = 1.0 - pwcca_distance_extended_original_anatome(x=b1_t, y=b2_t, backend='svd', use_layer_matrix='y')

print()
print(f'Google\'s: {pwcca_mean=}')
print(f'Google\'s (fixed): {pwcca_mean2=}')
print(f'Our code: {pwcca_ultimateanatome=}')
print(f'Our code: {pwcca_ultimateanatome_L1=}')
print(f'Our code: {pwcca_ultimateanatome_L2=}')
# print(f'Our code: {pwcca_extended_anatome=}')
# print(f'Our code: {pwcca_extended_anatome_L1=}')
# print(f'Our code: {pwcca_extended_anatome_L2=}')


print()
