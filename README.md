# imctoolkit

Python package for processing segmented multi-channel images

Documentation is available at [https://bodenmillergroup.github.io/imctoolkit](https://bodenmillergroup.github.io/imctoolkit)

## Requirements

This package requires Python 3.8 or later.

Python package dependencies are listed in [requirements.txt](https://github.com/BodenmillerGroup/imctoolkit/blob/main/requirements.txt).

Optional Python package dependencies include:

* [anndata](https://pypi.org/project/anndata) for exporting single-cell data as AnnData object
* [fcswrite](https://pypi.org/project/fcswrite) for writing single-cell data as FCS file
* [networkx](https://pypi.org/project/networkx) for exporting spatial cell graphs to networkx
* [python-igraph](https://pypi.org/project/python-igraph) for exporting spatial cell graphs to igraph
* [opencv-python](https://pypi.org/project/opencv-python) for faster image processing

Using virtual environments is strongly recommended.

## Installation

Install imctoolkit and its dependencies with:

    pip install imctoolkit

To also install optional dependencies (see above):

    pip install imctoolkit[all]

## Usage

See [Quickstart](https://bodenmillergroup.github.io/imctoolkit/quickstart.html)

## Authors

Jonas Windhager  
[jonas.windhager@uzh.ch](mailto:jonas.windhager@uzh.ch)

## Contributing

[Contributing](CONTRIBUTING.md)

## Changelog

[Changelog](CHANGELOG.md)

## License

[MIT](LICENSE.md)

## Contents

```{toctree}
quickstart
api_reference
CONTRIBUTING
CHANGELOG
LICENSE
```
