import numpy as np
import pandas as pd

from imctoolkit import ImageSingleCellData
from pathlib import Path


class TestImageSingleCellData:
    def test_data(self, data: ImageSingleCellData):
        assert data.image_width == 60
        assert data.image_height == 60
        assert data.num_channels == 5
        assert data.channel_names == ['Ag107', 'Pr141', 'Sm147', 'Eu153', 'Yb172']
        assert data.num_cells == 47

    def test_compute_cell_intensities(self, data: ImageSingleCellData):
        mean_cell_intensities = data.compute_cell_intensities(np.nanmean)
        expected_mean_channel_intensities = [0.12480281, 0.04633365, 0.03550479, 0.05638942, 0.53933499]
        assert all(np.abs(mean_cell_intensities.mean(axis=0) - expected_mean_channel_intensities) < 10e-6)

    def test_compute_cell_centroid_distances(self, data: ImageSingleCellData):
        cell_centroid_distances = data.compute_cell_centroid_distances(metric='euclidean')
        assert np.sum(cell_centroid_distances < 15) == 509

    def test_compute_cell_border_distances(self, data: ImageSingleCellData):
        cell_border_distances = data.compute_cell_border_distances()
        assert np.sum(cell_border_distances < 15) == 837

    def test_to_dataset(self, data: ImageSingleCellData):
        data.to_dataset(cell_properties=True, cell_channel_properties=True)

    def test_to_dataframe(self, data: ImageSingleCellData):
        df = data.to_dataframe(cell_properties=True, cell_channel_properties=True)
        assert df.shape == (47, 35)

    def test_to_anndata(self, data: ImageSingleCellData):
        ad = data.to_anndata(cell_properties=True, cell_channel_properties=True)
        assert ad.shape == (47, 5)
        assert len(ad.layers) == 6

    def test_write_csv(self, data: ImageSingleCellData, tmp_path: Path):
        csv_file_path = tmp_path / '20210305_NE_mockData1_s0_a1_ac_fullFiltered.csv'
        data.write_csv(csv_file_path, cell_properties=True, cell_channel_properties=True, index=False)
        df = pd.read_csv(csv_file_path)
        assert df.shape == (47, 35)

    def test_write_fcs(self, data: ImageSingleCellData, tmp_path: Path):
        fcs_file_path = tmp_path / '20210305_NE_mockData1_s0_a1_ac_fullFiltered.fcs'
        data.write_fcs(fcs_file_path, cell_properties=True, cell_channel_properties=True)
