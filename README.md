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

###  Representation Similarity

- [Raghu et al. NIPS2017 SVCCA](https://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability)
- [Marcos et al. NeurIPS2018 PWCCA](https://papers.nips.cc/paper/7815-insights-on-representational-similarity-in-neural-networks-with-canonical-correlation)
- [Kornblith et al. ICML2019 CKA](http://proceedings.mlr.press/v97/kornblith19a.html)
    
```python
from anatome import SimilarityHook
model = resnet18()
hook1 = SimilarityHook(model, "layer3.0.conv1")
hook2 = SimilarityHook(model, "layer3.0.conv2")
model.eval()
with torch.no_grad():
    model(data[0])
hook1.distance(hook2, size=8)
```
    
### Loss Landscape Visualization

- [Li et al. NeurIPS2018](https://papers.nips.cc/paper/7875-visualizing-the-loss-landscape-of-neural-nets)

```python
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