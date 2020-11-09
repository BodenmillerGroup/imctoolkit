"""Various utilities"""

import pandas as pd
import xarray as xr


def to_table(arr: xr.DataArray) -> pd.DataFrame:
    """Converts a two-dimensional :class:`xarray.DataArray` to a :class:`pandas.DataFrame` object

    :param arr: a two-dimensional :class:`xarray.DataArray`
    :return: a `pandas.DataFrame` representation of the array
    """
    if arr.ndim != 2:
        raise ValueError(f'Expected a two-dimensional array, got {arr.ndim}-dimensional array')
    return pd.DataFrame(
        data=arr.values,
        index=pd.Index(arr.coords[arr.dims[0]].values, name=arr.dims[0]),
        columns=pd.Index(arr.coords[arr.dims[1]].values, name=arr.dims[1])
    )
