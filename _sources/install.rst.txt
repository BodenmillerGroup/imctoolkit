Installation
============

Requirements
------------

This package requires `Python 3.8`_ or later. Using a virtual environment is strongly recommended.

Additionally, imctoolkit depends on:

* `imctools`_
* `numpy`_
* `pandas`_
* `scikit-image`_
* `scipy`_
* `tifffile`_
* `xarray`_
* `xtiff`_

Optional dependencies include:

* `anndata`_ for exporting single-cell data as AnnData object
* `fcswrite`_ for writing single-cell data as FCS file
* `networkx`_ for exporting spatial cell graphs as networkx graphs
* `python-igraph`_ for exporting spatial cell graphs as igraph graphs
* `opencv-python`_ for faster image processing

All dependencies are automatically installed using ``pip``.

.. _Python 3.8: https://www.python.org/
.. _imctools: https://pypi.org/project/imctools/
.. _numpy: https://pypi.org/project/numpy/
.. _pandas: https://pypi.org/project/pandas/
.. _scikit-image: https://pypi.org/project/scikit-image/
.. _scipy: https://pypi.org/project/scipy/
.. _tifffile: https://pypi.org/project/tifffile/
.. _xarray: https://pypi.org/project/xarray/
.. _xtiff: https://pypi.org/project/xtiff/
.. _anndata: https://pypi.org/project/anndata/
.. _fcswrite: https://pypi.org/project/fcswrite/
.. _networkx: https://pypi.org/project/networkx/
.. _python-igraph: https://pypi.org/project/python-igraph/
.. _opencv-python: https://pypi.org/project/opencv-python/


Installing imctoolkit
---------------------

In your virtual environment, install imctoolkit and its dependencies with::

    pip install git+https://github.com/BodenmillerGroup/imctoolkit

To also install optional dependencies, install imctoolkit with::

    pip install "git+https://github.com/BodenmillerGroup/imctoolkit#egg=imctoolkit[all]"


Installing imctoolkit for R
---------------------------

Please refer to `imctoolkitr`_ for instructions how to install imctoolkit for R.

.. _imctoolkitr: https://www.github.com/BodenmillerGroup/imctoolkitr
