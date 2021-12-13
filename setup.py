"""
conda create -n uanatome python=3.9
conda activate uanatome
conda remove --all --name uanatome
rm -rf /Users/brando/anaconda3/envs/uanatome
"""
from setuptools import find_packages, setup

import pathlib

# The directory containing this file
# HERE = pathlib.Path(__file__).parent
# here = os.path.abspath(os.path.dirname(__file__))
HERE = pathlib.Path('~/ultimate-anatome/').expanduser()

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='ultimate-anatome',
    version='0.1.1',
    description='Ἀνατομή (Anatome) is a PyTorch library to analyze representation of neural networks',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/brando90/ultimate-anatome',
    author='Brando Miranda',
    author_email='brandojazz@gmail.com',
    python_requires='>=3.9.0',
    license='MIT',
    packages=find_packages(),
    install_requires=['torch>=1.9.0',
                      'torchvision>=0.10.0',
                      'torchaudio>=0.9.1',
                      'tqdm'
                      ]
)
