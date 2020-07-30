from anatome import cca
import pytest
import torch


def test_cca_shape():
    i1 = torch.randn(10, 5)
    i2 = torch.randn(10, 8)
    i3 = torch.randn(10, 20)
    i4 = torch.randn(8, 5)
    cca.cca(i1, i1, 'svd')
    cca.cca(i1, i2, 'qr')
    with pytest.raises(ValueError):
        cca.cca(i1, i3, 'svd')
    with pytest.raises(ValueError):
        cca.cca(i1, i4, 'svd')
    with pytest.raises(ValueError):
        cca.cca(i1, i2, 'wrong')
