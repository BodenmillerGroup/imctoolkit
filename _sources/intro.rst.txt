Introduction
============

imctoolkit is a Python package for processing segmented multi-channel images.

Specific to IMC data processing, it bridges the `IMC segmentation pipeline`_ (without the single-cell measurement parts)
and downstream data analysis, without being restricted to specific frameworks.

For example, one could use imctoolkit for processing existing image/cell mask pairs obtained from external tools (e.g.
`Ilastik`_) to quickly analyzing single-cell data and spatial cell graphs in R.

An R wrapper package to use imctoolkit directly from within R is available from `imctoolkitr`_.

.. note::

    Most tasks enabled by imctoolkit can be achieved using other tools and frameworks. However, this package provides a
    common framework for straight-forward IMC data processing using Python, without to need to explicitly call external
    tools such as `CellProfiler`_.

.. _IMC segmentation pipeline: https://github.com/BodenmillerGroup/ImcSegmentationPipeline
.. _Ilastik: https://www.ilastik.org
.. _CellProfiler: https://cellprofiler.org
.. _imctoolkitr: https://www.github.com/BodenmillerGroup/imctoolkitr