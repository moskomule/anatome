import pytest
import torch
from torch import nn

from anatome import distance


@pytest.fixture
def matrices():
    return torch.randn(10, 5), torch.randn(10, 8), torch.randn(10, 20), torch.randn(8, 5)


@pytest.mark.parametrize('mat_size', ([10, 5], [100, 50], [500, 200]))
def test_cca_consistency(mat_size):
    def gen():
        x, y = torch.randn(*mat_size, dtype=torch.float64), torch.randn(*mat_size, dtype=torch.float64)
        x = distance._zero_mean(x, 0)
        y = distance._zero_mean(y, 0)
        return x, y

    x, y = gen()
    cca_svd = distance.cca_by_svd(x, y)
    cca_qr = distance.cca_by_qr(x, y)
    torch.testing.assert_close(cca_svd[0].abs(), cca_qr[0].abs())
    torch.testing.assert_close(cca_svd[1].abs(), cca_qr[1].abs())
    torch.testing.assert_close(cca_svd[2], cca_qr[2])


@pytest.mark.parametrize('method', ['svd', 'qr'])
def test_cca_shape(matrices, method):
    i1, i2, i3, i4 = matrices
    distance.cca(i1, i1, method)
    a, b, diag = distance.cca(i1, i2, method)
    assert list(a.size()) == [i1.size(1), diag.size(0)]
    assert list(b.size()) == [i2.size(1), diag.size(0)]
    with pytest.raises(ValueError):
        # needs more batch size
        distance.cca(i1, i3, method)
    with pytest.raises(ValueError):
        distance.cca(i1, i4, method)
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
    distance.orthogonal_procrustes_distance(i1, i3)
    with pytest.raises(ValueError):
        distance.orthogonal_procrustes_distance(i1, i4)


@pytest.mark.parametrize('method', ['pwcca', 'svcca', 'lincka', 'opd'])
def test_similarity_hook_linear(method):
    model1 = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 4))
    model2 = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 4))
    with pytest.raises(RuntimeError):
        distance.Distance(model1, model2, method=method, model1_names=['3'])

    dist = distance.Distance(model1, model2, method=method)

    assert dist.convert_names(model1, None, None, False) == ['0', '1']
    with torch.no_grad():
        dist.forward(torch.randn(13, 3))

    dist.between("1", "1")


@pytest.mark.parametrize('resize_by', ['avg_pool', 'dft'])
def test_similarity_hook_conv2d(resize_by):
    model1 = nn.Sequential(nn.Conv2d(3, 3, kernel_size=3), nn.Conv2d(3, 4, kernel_size=3))
    model2 = nn.Sequential(nn.Conv2d(3, 3, kernel_size=3), nn.Conv2d(3, 4, kernel_size=3))

    dist = distance.Distance(model1, model2, model1_names=['0', '1'], model2_names=['0', '1'], method='lincka')

    with torch.no_grad():
        dist.forward(torch.randn(13, 3, 11, 11))

    dist.between('1', '1', size=5)
    dist.between('1', '1', size=7)
    with pytest.raises(RuntimeError):
        dist.between('1', '1', size=8)

    dist.between('0', '1')
