#!/usr/bin/env python

from distutils.core import setup

setup(
    name='imctoolkit',
    version='0.1.0',
    description='IMC Toolkit',
    author='Jonas Windhager',
    author_email='jonas.windhager@uzh.ch',
    url='https://github.com/BodenmillerGroup/imctoolkit',
    packages=['imctoolkit'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    install_requires=[
        'imctools',
        'numpy',
        'pandas',
        'scikit-image',
        'scipy',
        'tifffile',
        'xarray',
        'xtiff',
    ],
    python_requires='>=3.8',
    extras_require={
        'all': ['anndata', 'networkx', 'python-igraph', 'opencv-python'],
    },
)
