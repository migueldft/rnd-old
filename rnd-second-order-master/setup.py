#!/usr/bin/env python

"""Define the setup function using setup.cfg."""
import io

from setuptools import setup, find_packages


REQUIRED_PACKAGES = [
    'category-encoders==2.0.0',
    'pycm==2.2',
    'joblib==0.13.2',
    'scikit-learn==0.21.3',
    'scikit-optimize==0.5.2',
    'seaborn==0.9.0',
    'xgboost==0.90',
    'Unidecode==1.1.1',
    'PyYAML==5.1.1',
    'Cerberus==1.3.1',
    'pandas==0.25.1',
    'awscli==1.16.234',
    'pyarrow==0.14.1',
    'dask[dataframe]==2.3.0',
    'sagemaker==1.39.2',
    'matplotlib==3.1.1',
    #Utils
    'pandas==0.25.1',
    'pyarrow==0.14.0',
    'dask[dataframe]==2.3.0',
    'loguru==0.4.1',
    'jsonschema==3.2.0',
    #API
    'falcon==2.0.0',
    'gevent==1.4.0',
    'greenlet==0.4.15',
    'gunicorn==19.9.0',
    'tornado==6.0.4',
]


LINKS = [
    'git+ssh://git@github.com/dafiti-group/rnd_data_libs.git@pipenv_plus_packaging=0.3.1',
]

# long description
def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


setup_dict = dict(
    name='second_order',
    version="1.2.0",
    description="R&D second order customer prediction.",
    url='https://github.com/dafiti-group/rnd-second-order',
    license='Apache 2.0',
    author="Ricardo Manhães Savii",
    author_email="ricardo.savii@dafiti.com.br",
    maintainer='Ricardo Manhães Savii',
    maintainer_email='ricardo.savii@dafiti.com.br',
    packages=find_packages(exclude=['tests']),
    install_requires=REQUIRED_PACKAGES,
    dependency_links=LINKS,
    platforms='Linux/MacOSX',

    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 0.4 - Beta',
        'Environment :: Other Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
)

if __name__ == '__main__':
    setup(**setup_dict)
