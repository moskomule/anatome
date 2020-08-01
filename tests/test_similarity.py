import pytest
import torch
from torch import nn

from anatome import similarity


def test_cca_shape():
    i1 = torch.randn(10, 5)
    i2 = torch.randn(10, 8)
    i3 = torch.randn(10, 20)
    i4 = torch.randn(8, 5)
    similarity.cca(i1, i1, 'svd')
    similarity.cca(i1, i2, 'qr')
    with pytest.raises(ValueError):
        similarity.cca(i1, i3, 'svd')
    with pytest.raises(ValueError):
        similarity.cca(i1, i4, 'svd')
    with pytest.raises(ValueError):
        similarity.cca(i1, i2, 'wrong')


def test_cka_shape():
    i1 = torch.randn(10, 5)
    i2 = torch.randn(10, 8)
    i3 = torch.randn(10, 20)
    i4 = torch.randn(8, 5)
    similarity.linear_cka_distance(i1, i2, True)
    similarity.linear_cka_distance(i1, i3, True)
    similarity.linear_cka_distance(i1, i2, False)
    with pytest.raises(ValueError):
        similarity.linear_cka_distance(i1, i4, True)


def test_similarity_hook_linear():
    model1 = nn.Linear(3, 3)
    model2 = nn.Linear(3, 5)
    hook1 = similarity.SimilarityHook(model1, '')
    hook2 = similarity.SimilarityHook(model2, '')
    input = torch.randn(13, 3)
    with torch.no_grad():
        model1(input)
        model2(input)

    hook1.distance(hook2)


def test_similarity_hook_conv2d():
    model1 = nn.Conv2d(3, 3, kernel_size=3)
    model2 = nn.Conv2d(3, 5, kernel_size=3)
    hook1 = similarity.SimilarityHook(model1, '')
    hook2 = similarity.SimilarityHook(model2, '')
    input = torch.randn(13, 3, 11, 11)
    with torch.no_grad():
        model1(input)
        model2(input)

    hook1.distance(hook2, size=7)

    with pytest.raises(RuntimeError):
        hook1.distance(hook2, size=19)
