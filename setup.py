from setuptools import find_packages, setup

import pathlib

# The directory containing this file
# HERE = pathlib.Path(__file__).parent
HERE = pathlib.Path('~/my_anatome/').expanduser()

# The text of the README file
README = (HERE / "README.md").read_text()

install_requires = [
    'torch>=1.9.0',
    'torchvision>=0.10.0',
    'tqdm'
]

setup(
    name='my_anatome',
    version='0.0.1',
    description='Ἀνατομή (Anatome) is a PyTorch library to analyze representation of neural networks',
    long_description=README,
    url='https://github.com/brando90/my_anatome',
    author='Ryuichiro Hataya and Brando Miranda',
    author_email='brandojazz@gmail.com',
    python_requires='>=3.9.0',
    license='MIT',
    packages=find_packages(),
    install_requires=install_requires
)
