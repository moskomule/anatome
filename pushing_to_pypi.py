# Steps to upload to pypi

```angular2html
pip install twine
```

go to project src and do:
```angular2html
python setup.py sdist bdist_wheel
```

create the distribution for pypi:
```angular2html
twine check dist/*
```

## Upload to pytest [optional]

```angular2html
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```
then click the url that appears. e.g.
```angular2html
Uploading distributions to https://test.pypi.org/legacy/
Enter your username: brando90
Enter your password: 
Uploading ultimate_utils-0.1.0-py3-none-any.whl
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 94.2k/94.2k [00:01<00:00, 52.4kB/s]
Uploading ultimate-utils-0.1.0.tar.gz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 82.5k/82.5k [00:01<00:00, 75.0kB/s]

View at:
https://test.pypi.org/project/ultimate-utils/0.1.0/
```

## Upload to pypi

```angular2html
twine upload dist/*
```
click url that appears to test it worked e.g.
```angular2html
Uploading distributions to https://upload.pypi.org/legacy/
Enter your username: brando90
Enter your password: 
Uploading ultimate_utils-0.1.0-py3-none-any.whl
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 94.2k/94.2k [00:02<00:00, 32.9kB/s]
Uploading ultimate-utils-0.1.0.tar.gz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 82.5k/82.5k [00:01<00:00, 43.4kB/s]

View at:
https://pypi.org/project/ultimate-utils/0.1.0/
```

then make sure you delete the build and dist:
```angular2html
rm -rf build dist
```
this avoids you accidentally trying to upload the same version of your package twice to pypi 
(which pypi won't let you do anyone but it will throw errors and perhaps confuse you).

## Test by pip installing it

create fresh conda env:
```angular2html
conda create -n test_env python=3.9
conda activate test_env
```

Test by pip installing it to your env:
```angular2html
pip install ultimate-utils
```

To test the installation uutils do:

```
python -c "import uutils; uutils.hello()"
python -c "import uutils; uutils.torch_uu.hello()"
```

it should print something like the following:

```

hello from uutils __init__.py in:
<module 'uutils' from '/Users/brando/ultimate-utils/ultimate-utils-proj-src/uutils/__init__.py'>


hello from torch_uu __init__.py in:
<module 'uutils.torch_uu' from '/Users/brando/ultimate-utils/ultimate-utils-proj-src/uutils/torch_uu/__init__.py'>

```

To test pytorch do:
```
python -c "import uutils; uutils.torch_uu.gpu_test_torch_any_device()"
```
To test if pytorch works with gpu do (it should fail if no gpus are available):
```
python -c "import uutils; uutils.torch_uu.gpu_test()"
```

### Setup.py

See the setup.py for ultimate utils to see a nice example. 
In particular reading the readme into the long description field is really nice to display the readme in pypi.
See: https://github.com/brando90/ultimate-utils/blob/master/ultimate-utils-proj-src/setup.py

e.g.

```
from setuptools import setup
from setuptools import find_packages

import pathlib

# The directory containing this file
# HERE = pathlib.Path(__file__).parent
HERE = pathlib.Path('~/ultimate-utils/').expanduser()

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='ultimate-utils',  # project name
    version='0.1.0',
    description='Brandos ultimate utils for science, machine learning and AI',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://pypi.org/project/ultimate-utils',
    author='Brando Miranda',
    author_email='brandojazz@gmail.com',
    # python_requires='>=3.9.0',
    license='MIT',
    packages=find_packages(),  # imports all modules (folder with __init__.py) & python files in this folder (since defualt args are . and empty exculde i.e. () )
    install_requires=['dill',
                      'networkx>=2.5',
                      'scipy',
                      'scikit-learn',
                      'lark-parser',
                      'torchtext',
                      'tensorboard',
                      'pandas',
                      'progressbar2',
                      'transformers',
                      'requests',
                      'aiohttp',
                      'numpy',
                      'plotly',
                      'wandb',
                      'matplotlib',
                      # 'torch'  # todo - try later

                      # 'pygraphviz'  # removing because it requires user to install graphviz and gives other issues, e.g. if the user does not want to do graph stuff then uutils shouldn't need to force the user to install uutils
                      ]
)
```

# Reference

https://realpython.com/pypi-publish-python-package/#publishing-to-pypi
