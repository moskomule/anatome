from setuptools import find_packages, setup

install_requires = [
    'torch>=1.10.0',
    'torchvision>=0.11.1',
]

setup(
    name='anatome',
    version='0.0.5',
    description='Ἀνατομή is a PyTorch library to analyze representation of neural networks',
    author='Ryuichiro Hataya',
    install_requires=install_requires,
    packages=find_packages()
)
