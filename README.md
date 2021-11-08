# anatome ![](https://github.com/moskomule/anatome/workflows/pytest/badge.svg)

Ἀνατομή is a PyTorch library to analyze internal representation of neural networks

This project is under active development and the codebase is subject to change.

**v0.0.5 introduces significant changes to `distance`.**

## Installation

`anatome` requires

```
Python>=3.9.0
PyTorch>=1.10
torchvision>=0.11
```

After the installation of PyTorch, install `anatome` as follows:

```
pip install -U git+https://github.com/moskomule/anatome
```

## Available Tools

### Representation Similarity

To measure the similarity of learned representation, `anatome.SimilarityHook` is a useful tool. Currently, the following
methods are implemented.

- [Raghu et al. NIPS2017 SVCCA](https://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability)
- [Marcos et al. NeurIPS2018 PWCCA](https://papers.nips.cc/paper/7815-insights-on-representational-similarity-in-neural-networks-with-canonical-correlation)
- [Kornblith et al. ICML2019 Linear CKA](http://proceedings.mlr.press/v97/kornblith19a.html)
- [Ding et al. arXiv Orthogonal Procrustes distance](https://arxiv.org/abs/2108.01661)

```python
import torch
from torchvision.models import resnet18
from anatome import Distance

random_model = resnet18()
learned_model = resnet18(pretrained=True)
distance = Distance(random_model, learned_model, method='pwcca')
with torch.no_grad():
    distance.forward(torch.randn(256, 3, 224, 224))

# resize if necessary by specifying `size`
distance.between("layer3.0.conv1", "layer3.0.conv1", size=8)
```

### Loss Landscape Visualization

- [Li et al. NeurIPS2018](https://papers.nips.cc/paper/7875-visualizing-the-loss-landscape-of-neural-nets)

```python
from torch.nn import functional as F
from torchvision.models import resnet18
from anatome import landscape2d

x, y, z = landscape2d(resnet18(),
                      data,
                      F.cross_entropy,
                      x_range=(-1, 1),
                      y_range=(-1, 1),
                      step_size=0.1)
imshow(z)
```

![](assets/landscape2d.svg)
![](assets/landscape3d.svg)

### Fourier Analysis

- Yin et al. NeurIPS 2019 etc.,

```python
from torch.nn import functional as F
from torchvision.models import resnet18
from anatome import fourier_map

map = fourier_map(resnet18(),
                  data,
                  F.cross_entropy,
                  norm=4)
imshow(map)
```

![](assets/fourier.svg)

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