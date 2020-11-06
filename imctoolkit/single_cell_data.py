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

from imctoolkit.multichannel_image import MultichannelImage

try:
    import anndata
except:
    anndata = None

try:
    import fcswrite
except:
    fcswrite = None


class SingleCellData:
    """Single-cell data extracted from a multi-channel image

    :ivar img: intensity image, as :class:`xarray.DataArray` with dimensions ``(c, y, x)`` and, optionally, channel
        names as coordinate for the ``c`` dimension
    :ivar mask: cell mask, as :class:`numpy.ndarray` of shape ``(y, x)``, where ``0`` indicates background pixels and
        non-zero pixels indicate the cell ID
    :ivar region_properties: list of :class:`RegionProperties` computed by the current instance, see :func:`regionprops`
    :ivar cell_ids: cell IDs, as one-dimensional :class:`numpy.ndarray`
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
        :type img: MultichannelImage, or any type supported by xarray.DataArrays
        :param mask: cell mask, shape: ``(y, x)``
        :type mask: any type supported by numpy.ndarrays
        :param channel_names: channel names
        :param region_properties: list of :class:`RegionProperties` to compute, defaults to
            :attr:`DEFAULT_REGION_PROPERTIES` when ``None``
        """
        if region_properties is None:
            region_properties = self.DEFAULT_REGION_PROPERTIES
        if isinstance(img, MultichannelImage):
            img = img.data
        if not isinstance(img, xr.DataArray):
            img = xr.DataArray(data=img, dims=('c', 'y', 'x'))
        mask = np.asarray(mask)
        if img.dims != ('c', 'y', 'x'):
            raise ValueError(f'Invalid image dimensions: expected ("c", "y", "x"), got {img.dims}')
        if channel_names is not None:
            img.coords['c'] = channel_names
        if mask.shape != img.shape[1:]:
            raise ValueError(f'Inconsistent mask {mask.shape} and image {img.shape[1:]} shapes')
        self.img = img
        self.mask = mask
        self.region_properties = list(region_properties)
        self.cell_ids = np.unique(mask[mask != 0])

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
    def channel_names(self) -> np.ndarray:
        """Channel names"""
        return self.img.coords['c'].values

    @property
    def num_cells(self) -> int:
        """Number of cells"""
        return len(self.cell_ids)

    @cached_property
    def min_intensities(self) -> xr.DataArray:
        """Minimum cell intensities

        :return: DataArray with coordinates ``(cell IDs, channel names)``
        """
        return self.compute_cell_intensities(np.nanmin)

    @property
    def min_intensities_table(self) -> pd.DataFrame:
        """Minimum cell intensities

        :return: DataFrame (index: cell IDs, columns: channel names)
        """
        return self._to_table(self.min_intensities)

    @cached_property
    def max_intensities(self) -> xr.DataArray:
        """Maximum cell intensities

        :return: DataArray with coordinates ``(cell IDs, channel names)``
        """
        return self.compute_cell_intensities(np.nanmax)

    @property
    def max_intensities_table(self) -> pd.DataFrame:
        """Maximum cell intensities

        :return: DataFrame (index: cell IDs, columns: channel names)
        """
        return self._to_table(self.max_intensities)

    @cached_property
    def mean_intensities(self) -> xr.DataArray:
        """Mean cell intensities

        :return: DataArray with coordinates ``(cell IDs, channel names)``
        """
        return self.compute_cell_intensities(np.nanmean)

    @property
    def mean_intensities_table(self) -> pd.DataFrame:
        """Mean cell intensities

        :return: DataFrame (index: cell IDs, columns: channel names)
        """
        return self._to_table(self.mean_intensities)

    @cached_property
    def median_intensities(self) -> xr.DataArray:
        """Median cell intensities

        :return: DataArray with coordinates ``(cell IDs, channel names)``
        """
        return self.compute_cell_intensities(np.nanmedian)

    @property
    def median_intensities_table(self) -> pd.DataFrame:
        """Median cell intensities

        :return: DataFrame (index: cell IDs, columns: channel names)
        """
        return self._to_table(self.median_intensities)

    @cached_property
    def std_intensities(self) -> xr.DataArray:
        """Standard deviations of cell intensities

        :return: DataArray with coordinates ``(cell IDs, channel names)``
        """
        return self.compute_cell_intensities(np.nanstd)

    @property
    def std_intensities_table(self) -> pd.DataFrame:
        """Standard deviations of cell intensities

        :return: DataFrame (index: cell IDs, columns: channel names)
        """
        return self._to_table(self.std_intensities)

    @cached_property
    def var_intensities(self) -> xr.DataArray:
        """Variances of cell intensities

        :return: DataArray with coordinates ``(cell IDs, channel names)``
        """
        return self.compute_cell_intensities(np.nanvar)

    @property
    def var_intensities_table(self) -> pd.DataFrame:
        """Variances of cell intensities

        :return: DataFrame (index: cell IDs, columns: channel names)
        """
        return self._to_table(self.var_intensities)

    @cached_property
    def regionprops(self) -> xr.DataArray:
        """Region properties

        For a list of computed properties, see :attr:`region_properties`.

        The properties ``'label'`` and ``'centroid'`` are always computed.

        :return: DataArray with coordinates ``(cell_id, property_name)``
        """
        properties = ['label', 'centroid'] + [rp.value for rp in self.region_properties]
        regionprops_dict = measure.regionprops_table(self.mask, properties=properties)
        df = pd.DataFrame(regionprops_dict, index=regionprops_dict.pop('label'))
        return xr.DataArray(data=df, dims=('cell', 'property'))

    @property
    def regionprops_table(self) -> pd.DataFrame:
        """Region properties

        For a list of computed properties, see :attr:`region_properties`.

        The properties ``'label'`` and ``'centroid'`` are always computed.

        :return: DataFrame (index: cell IDs, columns: regionprops property names)
        """
        return self._to_table(self.regionprops)

    @property
    def centroids(self) -> xr.DataArray:
        """Cell centroids

        :return: cell centroids, shape: ``(cells, dimensions=2)``
        """
        return self.regionprops.loc[:, ['centroid-0', 'centroid-1']]

    @staticmethod
    def _to_table(arr: xr.DataArray) -> pd.DataFrame:
        return pd.DataFrame(
            data=arr.values,
            index=pd.Index(arr.coords[arr.dims[0]].values, name=arr.dims[0]),
            columns=pd.Index(arr.coords[arr.dims[1]].values, name=arr.dims[1])
        )

    def compute_cell_intensities(self, aggr: Callable[[np.ndarray], Any]) -> xr.DataArray:
        """Compute cell intensity values

        :param aggr: function for aggregating the pixel values of a cell
        :return: DataArray with coordinates ``(cell IDs, channel names)``
        """
        arr = xr.DataArray(dims=('cell', 'channel'), coords={'cell': self.cell_ids, 'channel': self.channel_names})
        for channel_name in self.channel_names:
            channel_img = self.img.loc[channel_name].values
            arr.loc[:, channel_name] = [aggr(channel_img[self.mask == cell_id]) for cell_id in self.cell_ids]
        return arr

    def compute_centroid_distances(self, metric: str = 'euclidean') -> xr.DataArray:
        """Compute the pairwise distances between cell centroids

        :param metric: the distance metric to use, see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
        :return: symmetric centroid distance matrix
        """
        dist_mat = distance.squareform(distance.pdist(self.centroids.values, metric=metric))
        return xr.DataArray(data=dist_mat, dims=('cell_i', 'cell_j'),
                            coords={'cell_i': self.cell_ids, 'cell_j': self.cell_ids})

    def compute_border_distances(self) -> xr.DataArray:
        """Compute the pairwise Euclidean distances between cell borders

        :return: symmetric border distance matrix
        """
        # TODO speed up computation, e.g. by only computing distances between pixels belonging to cells
        dist_mat = np.zeros((self.num_cells, self.num_cells))
        cell_masks = [self.mask == cell_id for cell_id in self.cell_ids]
        for i, i_id in enumerate(self.cell_ids[:-1]):
            i_dist = distance_transform_edt(self.mask != i_id)
            dist_mat[i, (i + 1):] = [np.amin(i_dist[cell_masks[j]]) for j in range(i + 1, self.num_cells)]
        dist_mat += dist_mat.transpose()
        return xr.DataArray(data=dist_mat, dims=('cell_i', 'cell_j'),
                            coords={'cell_i': self.cell_ids, 'cell_j': self.cell_ids})

    def to_dataframe(self) -> pd.DataFrame:
        """Returns a :class:`pandas.DataFrame` representation of the current instance

        Column names for intensity attributes are prefixed according to the aggregation, e.g. columns from
        :attr:`mean_intensities_table` are included as ``'mean_{channel_name}'``.

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
        return xr.Dataset(data_vars={
            'min_intensities': self.min_intensities,
            'max_intensities': self.max_intensities,
            'mean_intensities': self.mean_intensities,
            'median_intensities': self.median_intensities,
            'std_intensities': self.std_intensities,
            'var_intensities': self.var_intensities,
            'regionprops': self.regionprops,
        })

    def to_anndata(self) -> 'anndata.AnnData':
        """Returns an :class:`anndata.AnnData` representation of the current instance

        :return: AnnData object, where intensity values are stored as layers and region properties are stored as
            observations
        """
        if anndata is None:
            raise RuntimeError('anndata is not installed')
        return anndata.AnnData(
            obs=pd.DataFrame(data=self.regionprops_table, index=pd.Index(data=self.cell_ids.astype(str), name='cell')),
            var=pd.DataFrame(index=pd.Index(data=self.channel_names, name='channel')),
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

    def write_fcs(self, path: Union[str, Path], **kwargs):
        """Writes an FCS file, see :func:`to_dataframe` for format specifications

        Uses :func:`fcswrite.write_fcs` for writing FCS 3.0 files.

        :param path: path to the .fcs file to be written
        :param kwargs: other arguments passed to :func:`fcswrite.write_fcs`
        """
        if fcswrite is None:
            raise RuntimeError('fcswrite is not installed')
        fcswrite.write_fcs(path, self.channel_names, self.to_dataframe().values, **kwargs)

