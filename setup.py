from setuptools import find_packages, setup

import pathlib

# The directory containing this file
# HERE = pathlib.Path(__file__).parent
HERE = pathlib.Path('~/ultimate-anatome/').expanduser()

# The text of the README file
README = (HERE / "README.md").read_text()

# install_requires = [
#     'torch>=1.9.0',
#     'torchvision>=0.10.0',
#     'tqdm'
# ]

setup(
    name='ultimate-anatome',
    version='0.0.4',
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
                      'tqdm'
                      ]
)
