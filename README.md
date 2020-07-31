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

- CCAs
    - Raghu et al. NIPS2017 SVCCA
    - Marcos et al. NeurIPS2018 PWCCA
    - [ ] Kornblith et al. ICML2019 CKA
- [ ] Fourier analysis


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