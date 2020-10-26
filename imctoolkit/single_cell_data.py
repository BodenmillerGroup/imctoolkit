import numpy as np
import pandas as pd
import xarray as xr

from enum import Enum
from functools import cached_property
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from scipy.spatial import distance
from skimage import measure
from typing import Any, Callable, Optional, Sequence, Union

from .image import Image

try:
    import anndata
except:
    anndata = None


class SingleCellData:
    """Single-cell data extracted from a multi-channel image

    :ivar img: intensity image, as :class:`xarray.DataArray` with dimensions ``(c, y, x)`` and, optionally, channel
        names as coordinate for the ``c`` dimension
    :ivar mask: cell mask, as integer-:class:`numpy.ndarray` of shape ``(y, x)``, where ``0`` indicates background
        pixels and non-zero pixels indicate the cell ID
    :ivar region_properties: list of :class:`RegionProperties` computed by the current instance, see :func:`regionprops`
        and :func:`regionprops_table`
    """

    class RegionProperties(Enum):
        """Enumeration of regionprops properties supported by this class, see
        https://scikit-image.org/docs/0.17.x/api/skimage.measure.html#skimage.measure.regionprops
        """
        AREA = 'area'
        BBOX = 'bbox'
        BBOX_AREA = 'bbox_area'
        CONVEX_AREA = 'convex_area'
        CONVEX_IMAGE = 'convex_image'
        COORDS = 'coords'
        ECCENTRICITY = 'eccentricity'
        EQUIVALENT_DIAMETER = 'equivalent_diameter'
        EULER_NUMBER = 'euler_number'
        EXTENT = 'extent'
        FILLED_AREA = 'filled_area'
        FILLED_IMAGE = 'filled_image'
        IMAGE = 'image'
        INERTIA_TENSOR = 'inertia_tensor'
        INERTIA_TENSOR_EIGVALS = 'inertia_tensor_eigvals'
        LOCAL_CENTROID = 'local_centroid'
        MAJOR_AXIS_LENGTH = 'major_axis_length'
        MINOR_AXIS_LENGTH = 'minor_axis_length'
        MOMENTS = 'moments'
        MOMENTS_CENTRAL = 'moments_central'
        MOMENTS_HU = 'moments_hu'
        MOMENTS_NORMALIZED = 'moments_normalized'
        ORIENTATION = 'orientation'
        PERIMETER = 'perimeter'
        SLICE = 'slice'
        SOLIDITY = 'solidity'

    DEFAULT_REGION_PROPERTIES = [
        RegionProperties.AREA,
        RegionProperties.ECCENTRICITY,
        RegionProperties.MAJOR_AXIS_LENGTH,
        RegionProperties.MINOR_AXIS_LENGTH,
        RegionProperties.ORIENTATION,
    ]  #: List of :class:`RegionProperties` computed by default, see :func:`__init__`

    def __init__(self, img, mask, channel_names: Optional[Sequence[str]] = None,
                 region_properties: Optional[Sequence[RegionProperties]] = None):
        """

        :param img: intensity image, shape: ``(c, y, x)``
        :type img: Image, or any type supported by xarray.DataArrays
        :param mask: cell mask, shape: ``(y, x)``
        :type mask: any type supported by integer-numpy.ndarrays
        :param channel_names: channel names
        :param region_properties: list of :class:`RegionProperties` to compute, defaults to
            :attr:`DEFAULT_REGION_PROPERTIES` when ``None``
        """
        if isinstance(img, Image):
            img = img.data
        if not isinstance(img, xr.DataArray):
            img = xr.DataArray(img, dims=('c', 'y', 'x'))
        mask = np.asarray(mask, dtype='int')
        if img.dims != ('c', 'y', 'x'):
            raise ValueError(f'Invalid image dimensions: expected ("c", "y", "x"), got {img.dims}')
        if channel_names is not None:
            img.coords['c'] = channel_names
        if mask.shape != img.shape[1:]:
            raise ValueError(f'Inconsistent mask {mask.shape} and image {img.shape[1:]} shapes')
        self.img = img
        self.mask = mask
        self.region_properties = list(region_properties) or self.DEFAULT_REGION_PROPERTIES

    @property
    def image_width(self) -> int:
        """Image width in pixels"""
        return self.img.sizes['x']

    @property
    def image_height(self) -> int:
        """Image height in pixels"""
        return self.img.sizes['y']

    @property
    def num_channels(self) -> int:
        """Number of channels"""
        return self.img.sizes['c']

    @property
    def channel_names(self) -> Sequence[str]:
        """Channel names"""
        return self.img.coords['c']

    @property
    def num_cells(self) -> int:
        """Number of cells"""
        return len(self.cell_ids)

    @cached_property
    def cell_ids(self) -> Sequence[int]:
        """Cell IDs"""
        return np.unique(self.mask[self.mask != 0])

    @cached_property
    def min_intensities(self) -> xr.DataArray:
        """Minimum cell intensities

        :return: DataArray with coordinates ``(cell IDs, channel names)``
        """
        return self.compute_cell_intensity(np.nanmin)

    @property
    def min_intensities_table(self) -> pd.DataFrame:
        """Minimum cell intensities

        :return: DataFrame (index: cell IDs, columns: channel names)
        """
        return pd.DataFrame(self.min_intensities, index=self.cell_ids, columns=self.channel_names)

    @cached_property
    def max_intensities(self) -> xr.DataArray:
        """Maximum cell intensities

        :return: DataArray with coordinates ``(cell IDs, channel names)``
        """
        return self.compute_cell_intensity(np.nanmax)

    @property
    def max_intensities_table(self) -> pd.DataFrame:
        """Maximum cell intensities

        :return: DataFrame (index: cell IDs, columns: channel names)
        """
        return pd.DataFrame(self.max_intensities, index=self.cell_ids, columns=self.channel_names)

    @cached_property
    def mean_intensities(self) -> xr.DataArray:
        """Mean cell intensities

        :return: DataArray with coordinates ``(cell IDs, channel names)``
        """
        return self.compute_cell_intensity(np.nanmean)

    @property
    def mean_intensities_table(self) -> pd.DataFrame:
        """Mean cell intensities

        :return: DataFrame (index: cell IDs, columns: channel names)
        """
        return pd.DataFrame(self.mean_intensities, index=self.cell_ids, columns=self.channel_names)

    @cached_property
    def median_intensities(self) -> xr.DataArray:
        """Median cell intensities

        :return: DataArray with coordinates ``(cell IDs, channel names)``
        """
        return self.compute_cell_intensity(np.nanmedian)

    @property
    def median_intensities_table(self) -> pd.DataFrame:
        """Median cell intensities

        :return: DataFrame (index: cell IDs, columns: channel names)
        """
        return pd.DataFrame(self.median_intensities, index=self.cell_ids, columns=self.channel_names)

    @cached_property
    def std_intensities(self) -> xr.DataArray:
        """Standard deviations of cell intensities

        :return: DataArray with coordinates ``(cell IDs, channel names)``
        """
        return self.compute_cell_intensity(np.nanstd)

    @property
    def std_intensities_table(self) -> pd.DataFrame:
        """Standard deviations of cell intensities

        :return: DataFrame (index: cell IDs, columns: channel names)
        """
        return pd.DataFrame(self.std_intensities, index=self.cell_ids, columns=self.channel_names)

    @cached_property
    def var_intensities(self) -> xr.DataArray:
        """Variances of cell intensities

        :return: DataArray with coordinates ``(cell IDs, channel names)``
        """
        return self.compute_cell_intensity(np.nanvar)

    @property
    def var_intensities_table(self) -> pd.DataFrame:
        """Variances of cell intensities

        :return: DataFrame (index: cell IDs, columns: channel names)
        """
        return pd.DataFrame(self.var_intensities, index=self.cell_ids, columns=self.channel_names)

    @property
    def regionprops(self) -> xr.DataArray:
        """Region properties

        For a list of computed properties, see :attr:`region_properties`.

        The properties ``'label'`` and ``'centroid'`` are always computed.

        :return: DataArray with coordinates ``(cell_id, property_name)``
        """
        return xr.DataArray(self.regionprops_table, dims=('cell', 'property'))

    @cached_property
    def regionprops_table(self) -> pd.DataFrame:
        """Region properties

        For a list of computed properties, see :attr:`region_properties`.

        The properties ``'label'`` and ``'centroid'`` are always computed.

        :return: DataFrame (index: cell IDs, columns: regionprops property names)
        """
        properties = ['label', 'centroid'] + [rp.value for rp in self.region_properties]
        df = pd.DataFrame(data=measure.regionprops_table(self.mask, properties=properties))
        df.rename(columns={'label': 'cell'})
        df.set_index('cell', inplace=True)
        return df

    def compute_cell_intensity(self, f: Callable[[np.ndarray], Any]) -> xr.DataArray:
        """Compute cell intensity values

        :param f: function for aggregating the pixel values of a cell
        :return: DataArray with coordinates ``(cell IDs, channel names)``
        """
        cell_intensities = xr.DataArray(dims=('cell', 'channel'),
                                        coords={'cell': self.cell_ids, 'channel': self.channel_names})
        for channel_name in self.channel_names:
            channel_img = self.img.data[channel_name]
            cell_intensities.loc[:, channel_name] = [f(channel_img[self.mask == cell_id]) for cell_id in self.cell_ids]
        return cell_intensities

    def compute_centroid_distances(self, metric: str = 'euclidean') -> xr.DataArray:
        """Compute the pairwise dist_mat between cell centroids

        :param metric: the distance metric to use, see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
        :return: symmetric centroid distance matrix
        """
        centroid_cols = [col for col in self.regionprops_table.columns if col.startswith('centroid')]
        centroids = self.regionprops_table.loc[:, centroid_cols].to_numpy()
        dist_mat = distance.squareform(distance.pdist(centroids, metric=metric))
        return xr.DataArray(dist_mat, dims=('cell', 'cell'), coords={'cell': self.cell_ids})

    def compute_border_distances(self) -> xr.DataArray:
        """Compute the pairwise dist_mat between cell borders

        :return: symmetric border distance matrix
        """
        dist_mat = xr.DataArray(dims=('cell', 'cell'), coords={'cell': self.cell_ids})
        for i, i_id in enumerate(self.cell_ids):
            for j_id in self.cell_ids[(i + 1):]:
                a = self.mask[self.mask != j_id]  # cell j is 0, everything else is 1
                a = distance_transform_edt(a, return_distances=True)  # dist_mat from non-zeros to the nearest 0
                dist_mat.loc[i_id, j_id] = dist_mat.loc[j_id, i_id] = np.amin(a[self.mask == i_id])
        return dist_mat

    def to_dataframe(self) -> pd.DataFrame:
        """Returns a :class:`pandas.DataFrame` representation of the current instance

        Column names for intensity attributes are prefixed according to the aggregation, e.g. columns from
        :attr:`mean_intensities_table` will be included as ``'mean_{channel_name}'``.

        Columns from :attr:`regionprops_table` are included without a prefix.

        :return: DataFrame (index: cell IDs, columns: see description)
        """
        df = pd.DataFrame(index=pd.Index(self.cell_ids, name='cell'))
        df = pd.merge(df, self.min_intensities_table.add_prefix('min_'), left_index=True, right_index=True)
        df = pd.merge(df, self.max_intensities_table.add_prefix('max'), left_index=True, right_index=True)
        df = pd.merge(df, self.mean_intensities_table.add_prefix('mean_'), left_index=True, right_index=True)
        df = pd.merge(df, self.median_intensities_table.add_prefix('median_'), left_index=True, right_index=True)
        df = pd.merge(df, self.std_intensities_table.add_prefix('std_'), left_index=True, right_index=True)
        df = pd.merge(df, self.var_intensities_table.add_prefix('var_'), left_index=True, right_index=True)
        df = pd.merge(df, self.regionprops_table, left_index=True, right_index=True)
        df.columns.name = 'feature'
        return df

    def to_dataset(self) -> xr.Dataset:
        """Returns an :class:`xarray.Dataset` representation of the current instance

        :return: Dataset containing intensity values and region properties, with cell ID, channel name and property name
            as coordinates
        """
        return xr.Dataset(
            data_vars={
                'min_intensities': xr.DataArray(self.min_intensities, dims=('cell', 'channel')),
                'max_intensities': xr.DataArray(self.max_intensities, dims=('cell', 'channel')),
                'mean_intensities': xr.DataArray(self.mean_intensities, dims=('cell', 'channel')),
                'median_intensities': xr.DataArray(self.median_intensities, dims=('cell', 'channel')),
                'std_intensities': xr.DataArray(self.std_intensities, dims=('cell', 'channel')),
                'var_intensities': xr.DataArray(self.var_intensities, dims=('cell', 'channel')),
                'regionprops': xr.DataArray(self.regionprops, dims=('cell', 'property')),
            },
            coords={'cell': self.cell_ids, 'channel': self.channel_names, 'property': self.region_properties},
        )

    def to_anndata(self) -> 'anndata.AnnData':
        """Returns an :class:`anndata.AnnData` representation of the current instance

        :return: AnnData object, where intensity values are stored as layers and region properties are stored as
            multi-dimensional observations
        """
        if anndata is None:
            raise RuntimeError('anndata is not installed')
        return anndata.AnnData(
            obs={'cell': self.cell_ids},
            var={'channel': self.channel_names},
            obsm=self.regionprops_table,
            layers={
                'min_intensities': self.min_intensities.values,
                'max_intensities': self.max_intensities.values,
                'mean_intensities': self.mean_intensities.values,
                'median_intensities': self.median_intensities.values,
                'std_intensities': self.std_intensities.values,
                'var_intensities': self.var_intensities.values,
            },
            shape=(self.num_cells, self.num_channels)
        )

    def write_csv(self, path: Union[str, Path], **kwargs):
        """Writes a CSV file, see :func:`to_dataframe` for format specifications

        :param path: path to the .csv file to be written
        :param kwargs: other arguments passed to :func:`pandas.DataFrame.to_csv`
        """
        self.to_dataframe().to_csv(path, **kwargs)
