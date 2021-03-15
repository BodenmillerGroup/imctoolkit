from imctoolkit import SpatialCellGraph


class TestSpatialCellGraph:
    def test_to_dataset(self, knn_graph: SpatialCellGraph):
        knn_graph.to_dataset()

    def test_to_networkx(self, knn_graph: SpatialCellGraph):
        g = knn_graph.to_networkx()
        assert len(g.nodes) == 47
        assert len(g.edges) == 235

    def test_to_igraph(self, knn_graph: SpatialCellGraph):
        g = knn_graph.to_igraph()
        assert len(g.vs) == 47
        assert len(g.es) == 235

    def test_load_dataset(self, knn_graph: SpatialCellGraph):
        ds = knn_graph.to_dataset()
        knn_graph = SpatialCellGraph.load_dataset(ds)
        assert knn_graph.num_cells == 47
        assert knn_graph.num_features == 35
        assert knn_graph.adj_mat.sum() == 235

    def test_construct_knn_graph(self, knn_graph: SpatialCellGraph):
        assert not knn_graph.is_undirected
        assert knn_graph.num_cells == 47
        assert knn_graph.num_features == 35
        assert knn_graph.adj_mat.sum() == 235

    def test_construct_dist_graph(self, dist_graph: SpatialCellGraph):
        assert dist_graph.is_undirected
        assert dist_graph.num_cells == 47
        assert dist_graph.num_features == 35
        assert dist_graph.adj_mat.sum() == 462
