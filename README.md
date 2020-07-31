# anatome ![](https://github.com/moskomule/anatome/workflows/pytest/badge.svg)

Ἀνατομή is a PyTorch library to analyze internal representation of neural networks

This project is under (hopefully) active development and the codebase is subject to change.

## Installation

anatome requires

```
Python>=3.8.0
PyTorch>=1.6.0
torchvision>=0.7.0
```

To install anatome, run

```
pip install -U git+https://github.com/moskomule/anatome
```

## Available Tools

###  CCAs: Compare representation of modules

- Raghu et al. NIPS2017 SVCCA
- Marcos et al. NeurIPS2018 PWCCA
- [ ] Kornblith et al. ICML2019 CKA
    
```python
from anatome import CCAHook
model = resnet18()
hook1 = CCAHook(model, "layer3.0.conv1")
hook2 = CCAHook(model, "layer3.0.conv2")
model.eval()
with torch.no_grad():
    model(torch.randn(120, 3, 224, 224))
hook1.distance(hook2, size=8)
```
    
### Loss Landscape Visualization

- Li et al. NeurIPS2018 ([Original Implementation](https://github.com/tomgoldstein/loss-landscape))

```python
from anatome import landscape1d
x, y = landscape1d(resnet18(),
                   data,
                   F.cross_entropy,
                   x_range=(-1, 1),
                   step_size=0.1)
plot(x, y, ...)
```

### Fourier Analysis

- Yin et al. NeurIPS 2019 etc.,

```python
from anatome import fourier_map
map = fourier_map(resnet18(),
                  data,
                  F.cross_entropy,
                  norm=4)
imshow(map)
```

## Citation

If you use this implementation in your research, please cite as:

```
@software{hataya2020anatome,
    author={Ryuichiro Hataya},
    title={anatome, a PyTorch library to analyze internal representation of neural networks},
    url={https://github.com/moskomule/anatome},
    year={2020}
}
```