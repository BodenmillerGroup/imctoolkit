import numpy as np
import pandas as pd
import xarray as xr

from typing import Sequence

from .single_cell_data import SingleCellData

try:
    import networkx as nx
except:
    nx = None

try:
    import igraph
except:
    igraph = None


class SpatialCellGraph:
    """Spatial cell graph constructed from single-cell data

    :ivar data: single-cell data, as :class:`pandas.DataFrame`, with cell IDs as index and feature names as columns
    :ivar adj_mat: boolean adjacency matrix, as :class:`xarray.DataArray` with coordinates ``(cell IDs, cell IDs)``
    """

    def __init__(self, data, adj_mat, _skip_check_params: bool = False):
        """

        :param data: single-cell data (rows: cell IDs, columns: feature names)
        :type data: SingleCellData, or any type supported by pandas.DataFrames
        :param adj_mat: boolean adjacency matrix, shape: ``(cells, features)``
        :type adj_mat: any type supported by xarray.DataArrays
        """
        if not _skip_check_params:
            data, adj_mat = self._check_params(data, adj_mat, 'adjacency')
        self.data = data
        self.adj_mat = adj_mat

    @staticmethod
    def _check_params(data, mat, mat_type: str):
        if isinstance(data, SingleCellData):
            data = data.to_dataframe()
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        if not isinstance(mat, xr.DataArray):
            mat = xr.DataArray(np.asarray(mat), dims=('cell', 'cell'))
        mat_cells_in_data = np.in1d(mat.coords['cell'], data.index)
        if not np.all(mat_cells_in_data):
            missing_cell_ids = mat.coords["cell"][mat_cells_in_data].values.tolist()
            raise ValueError(f'Missing cell data for cell IDs in {mat_type} matrix: {missing_cell_ids}')
        return data, mat

    @property
    def num_cells(self) -> int:
        """Number of cells"""
        return len(self.cell_ids)

    @property
    def cell_ids(self) -> Sequence[int]:
        """Cell IDs"""
        return self.data.index

    @property
    def num_features(self) -> int:
        """Number of features"""
        return len(self.feature_names)

    @property
    def feature_names(self) -> Sequence[str]:
        """Feature names"""
        return self.data.columns

    @property
    def is_undirected(self) -> bool:
        """``True``, if :attr:`adj_mat` is symmetric, ``False`` otherwise"""
        return np.allclose(self.adj_mat, self.adj_mat.transpose())

    def to_dataset(self) -> xr.Dataset:
        """Returns an :class:`xarray.Dataset` representation of the current instance

        :return: Dataset with :attr:`data` and :attr:`adj_mat` as members and ``(cell IDs, feature names)`` as
            coordinates
        """
        return xr.Dataset(data_vars={
            'data': xr.DataArray(self.data, dims=('cell', 'feature')),
            'adj_mat': xr.DataArray(self.adj_mat, dims=('cell', 'cell')),
        }, coords={'cell': self.cell_ids, 'feature': self.feature_names})

    def to_networkx(self, create_using=None) -> 'nx.Graph':
        """Returns a :class:`networkx.Graph` representation of the current instance

        :param create_using: type of graph to create, defaults to :class:`networkx.Graph` for undirected graphs and
            :class:`networkx.DiGraph` for directed graphs when ``None``
        :type create_using: see :func:`networkx.from_numpy_array`
        :return: Graph or DiGraph with cell IDs as node labels and features as node attributes
        """
        if nx is None:
            raise RuntimeError('networkx is not installed')
        if create_using is None:
            create_using = nx.Graph if self.is_undirected else nx.DiGraph
        graph = nx.from_numpy_array(self.adj_mat.values, create_using=create_using)
        graph = nx.relabel_nodes(graph, mapping=dict(zip(graph, self.cell_ids)), copy=False)
        node_attributes = {}
        for cell_id in self.cell_ids:
            node_attributes[cell_id] = self.data.loc[cell_id, :].to_dict()
        nx.set_node_attributes(graph, node_attributes)
        return graph

    def to_igraph(self, mode=None) -> 'igraph.Graph':
        """Returns an :class:`igraph.Graph` representation of the current instance

        :param mode: graph mode, defaults to :attr:`igraph.ADJ_UNDIRECTED` for undirected graphs and
            :attr:`igraph.ADJ_DIRECTED` for directed graphs when ``None``
        :type mode: see :func:`igraph.Graph.Adjacency`
        :return: Graph with cell IDs and features as vertex attributes
        """
        if igraph is None:
            raise RuntimeError('python-igraph is not installed')
        if mode is None:
            mode = igraph.ADJ_UNDIRECTED if self.is_undirected else igraph.ADJ_DIRECTED
        graph: igraph.Graph = igraph.Graph.Adjacency(self.adj_mat.values.tolist(), mode=mode)
        graph.vs['cell_id'] = self.cell_ids
        for feature_name in self.feature_names:
            graph.vs[feature_name] = self.data.loc[self.cell_ids, feature_name].values.tolist()
        return graph

    @staticmethod
    def load_dataset(dataset: xr.Dataset) -> 'SpatialCellGraph':
        """Creates a new :class:`SpatialCellGraph` from its dataset representation, see

        :param dataset: Dataset of the same format as created by :func:`to_dataset`
        :return: a new :class:`SpatialCellGraph` instance
        """
        return SpatialCellGraph(dataset['data'].to_dataframe(), dataset['adj_mat'])

    @classmethod
    def construct_knn_graph(cls, data, dist_mat, k: int) -> 'SpatialCellGraph':
        """Constructs a new k-nearest cell neighbor graph

        :param data: single-cell data (rows: cell IDs, columns: feature names)
        :type data: SingleCellData, or any type supported by pandas.DataFrames
        :param dist_mat: distance matrix, shape: ``(cells, features)``
        :type dist_mat: any type supported by xarray.DataArrays
        :param k: number of nearest neighbors for the graph construction
        :return: a directed k-nearest cell neighbor graph
        """
        data, dist_mat = cls._check_params(data, dist_mat, 'distance')
        adj_mat = xr.zeros_like(dist_mat, dtype='bool')
        knn_indices = np.argpartition(dist_mat.values, k + 1, axis=1)[:, :(k + 1)]
        for current_index, current_knn_indices in enumerate(knn_indices):
            adj_mat[current_index, current_knn_indices] = True
        adj_mat = np.fill_diagonal(adj_mat, False)
        return SpatialCellGraph(data, adj_mat, _skip_check_params=True)

    @classmethod
    def construct_dist_graph(cls, data, dist_mat, dist_thres: float) -> 'SpatialCellGraph':
        """Constructs a new cell neighborhood graph by distance thresholding

        :param data: single-cell data (rows: cell IDs, columns: feature names)
        :type data: SingleCellData, or any type supported by pandas.DataFrames
        :param dist_mat: distance matrix, shape: ``(cells, features)``
        :type dist_mat: any type supported by xarray.DataArrays
        :param dist_thres: distance hot_pixel_thres, (strictly) below which cells are considered neighbors
        :return: an undirected cell neighborhood graph
        """
        data, dist_mat = cls._check_params(data, dist_mat, 'distance')
        adj_mat = (dist_mat < dist_thres)
        adj_mat = np.fill_diagonal(adj_mat, False)
        return SpatialCellGraph(data, adj_mat, _skip_check_params=True)
