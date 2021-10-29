import pytest
import torch
from torch import nn

from anatome import distance


@pytest.fixture
def matrices():
    return torch.randn(10, 5), torch.randn(10, 8), torch.randn(10, 20), torch.randn(8, 5)


def test_cca_shape(matrices):
    i1, i2, i3, i4 = matrices
    distance.cca(i1, i1, 'svd')
    distance.cca(i1, i2, 'qr')
    with pytest.raises(ValueError):
        # needs more batch size
        distance.cca(i1, i3, 'svd')
    with pytest.raises(ValueError):
        distance.cca(i1, i4, 'svd')
    with pytest.raises(ValueError):
        distance.cca(i1, i2, 'wrong')


def test_cka_shape(matrices):
    i1, i2, i3, i4 = matrices
    distance.linear_cka_distance(i1, i2, True)
    distance.linear_cka_distance(i1, i3, True)
    distance.linear_cka_distance(i1, i2, False)
    with pytest.raises(ValueError):
        distance.linear_cka_distance(i1, i4, True)


def test_opd(matrices):
    i1, i2, i3, i4 = matrices
    distance.orthogonal_procrustes_distance(i1, i1)
    distance.orthogonal_procrustes_distance(i1, i2)
    with pytest.raises(ValueError):
        distance.orthogonal_procrustes_distance(i1, i4)


def test_similarity_hook_linear():
    model1 = nn.Linear(3, 3)
    model2 = nn.Linear(3, 5)
    hook1 = distance.DistanceHook(model1, '')
    hook2 = distance.DistanceHook(model2, '')
    input = torch.randn(13, 3)
    with torch.no_grad():
        model1(input)
        model2(input)

    hook1.distance(hook2)


@pytest.mark.parametrize('resize_by', ['avg_pool', 'dft'])
def test_similarity_hook_conv2d(resize_by):
    model1 = nn.Conv2d(3, 3, kernel_size=3)
    model2 = nn.Conv2d(3, 5, kernel_size=3)
    hook1 = distance.DistanceHook(model1, '')
    hook2 = distance.DistanceHook(model2, '')
    input = torch.randn(13, 3, 11, 11)
    with torch.no_grad():
        model1(input)
        model2(input)

    hook1.distance(hook2, size=7, downsample_method=resize_by)

    with pytest.raises(RuntimeError):
        hook1.distance(hook2, size=19, downsample_method=resize_by)
