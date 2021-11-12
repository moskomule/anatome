"""
Some experiment results seem funny with PWCCA so let's make sure the CCA values for of ultimate anatome matches or is
close to values of other code.

conda install pytorch torchvision torchaudio -c pytorch

conda install pytorch torchvision torchaudio -c pytorch==1.10
"""
#%%
import sys
from pprint import pprint

import torch
from matplotlib import pyplot as plt

from uutils.torch_uu.metrics.cca import cca_core

from torch import Tensor
# from anatome.similarity import svcca_distance, cca_by_svd, cca_by_qr, _compute_cca_traditional_equation
from anatome.distance import svcca_distance, cca_by_svd, cca_by_qr
import numpy as np
import random
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
# print(f'{anatome.__version__=}')
# import subprocess
# print(subprocess.check_output(['ls', '-l']))

print('\n-- start cca test --')


# tutorial shapes (500, 10000) (500, 10000) based on MNIST with 500 neurons from a FCNN
D, N = 500, 10_000

# - creating a random baseline
# b1 = np.random.randn(*acts1.shape)
# b2 = np.random.randn(*acts2.shape)
b1 = np.random.randn(D, N)
b2 = np.random.randn(D, N)
print('-- reproducibity finger print')
print(f'{b1.sum()=}')
print(f'{b2.sum()=}')
print(f'{b1.shape=}')

# - get cca values for baseline
print("\n-- Google's SVCCA -- ")
baseline = cca_core.get_cca_similarity(b1, b2, epsilon=1e-10, verbose=False)
# _plot_helper(baseline["cca_coef1"], "CCA coef idx", "CCA coef value")
# print("Baseline Mean CCA similarity", np.mean(baseline["cca_coef1"]))
# print("Baseline CCA similarity", baseline["cca_coef1"])
print(f'{len(baseline["cca_coef1"])=}')
print("Baseline CCA similarity", baseline["cca_coef1"][:6])
print(f"{np.mean(baseline['cca_coef1'])=}")

# - get sklern's cca's https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html
# from sklearn.cross_decomposition import CCA
# # cca = CCA(n_components=D)
# cca = CCA(n_components=6)
# cca.fit(b1, b2)

# -
print("\n-- Ultimate Anatome's SVCCA --")
# 'svcca': partial(svcca_distance, accept_rate=0.99, backend='svd')
# svcca_dist: Tensor = svcca_distance(x, y, accept_rate=0.99, backend='svd')
b1_t, b2_t = torch.from_numpy(b1), torch.from_numpy(b2)
# svcca: Tensor = 1.0 - svcca_distance(x=b1_t, y=b2_t, accept_rate=1.0, backend='svd')
# diag: Tensor = svcca_distance(x=b1_t, y=b2_t, accept_rate=0.99, backend='svd')
# a, b, diag = cca(x, y, backend='svd')
a, b, diag = cca_by_svd(b1_t, b2_t)
# a, b, diag = cca_by_qr(b1_t, b2_t)
# diag = _compute_cca_traditional_equation(b1_t, b2_t)
print(f'{diag.size()=}')
print(f'{diag[:6]=}')
print(f'{diag.mean()=}')
# print(f'{svcca=}')

print()
