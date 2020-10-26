Introduction
============

imctoolkit is a Python package for common tasks in processing segmented multi-channel images. As such, it bridges e.g.
the `IMC segmentation pipeline`_ (without the single-cell measurement parts) and downstream data analysis, without being
restricted to this specific component.

For example, one can use existing image/cell mask pairs to quickly extract single-cell data and spatial cell graphs and
export them to data structures for downstream analysis in R.

.. note::

    Most tasks enabled by imctoolkit can be achieved using other tools and frameworks. However, this package provides a
    common framework for straight-forward IMC data processing using Python, without to need to explicitly call external
    tools such as CellProfiler.

.. _IMC segmentation pipeline: https://github.com/BodenmillerGroup/ImcSegmentationPipeline

