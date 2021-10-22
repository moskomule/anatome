from setuptools import find_packages, setup

install_requires = [
    'torch>=1.9.0',
    'torchvision>=0.10.0',
    'tqdm'
]

setup(
    name='anatome',
    version='0.0.4',
    description='Ἀνατομή is a PyTorch library to analyze representation of neural networks',
    author='Ryuichiro Hataya',
    install_requires=install_requires,
    packages=find_packages()
)
