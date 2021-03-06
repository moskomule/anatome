from setuptools import setup, find_packages

install_requires = [
    'torch>=1.6.0',
    'torchvision>=0.7.0',
    'tqdm'
]

setup(
    name='anatome',
    version='0.0.1',
    description='Ἀνατομή is a PyTorch library to analyze representation of neural networks',
    author='Ryuichiro Hataya',
    author_email='hataya@keio.jp',
    install_requires=install_requires,
    packages=find_packages()
)
