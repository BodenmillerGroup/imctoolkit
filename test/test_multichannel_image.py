import pandas as pd
import pytest

from imctoolkit import MultichannelImage
from pathlib import Path


class TestMultichannelImage:
    def test_read_invalid_suffix(self):
        with pytest.raises(ValueError):
            MultichannelImage.read('file.unsupported_suffix')

    def test_read_imc_txt(self, raw_path: Path):
        txt_file_path = raw_path / '20210305_NE_mockData1' / '20210305_NE_mockData1_ROI_001_1.txt'
        img = MultichannelImage.read_imc_txt(txt_file_path)
        assert img.width == 60
        assert img.height == 60
        assert img.num_channels == 5
        assert img.channel_names == ['Ag107', 'Pr141', 'Sm147', 'Eu153', 'Yb172']

    def test_read_imc_mcd(self, raw_path: Path):
        mcd_file_path = raw_path / '20210305_NE_mockData1' / '20210305_NE_mockData1.mcd'
        img = MultichannelImage.read_imc_mcd(mcd_file_path, 1)
        assert img.width == 60
        assert img.height == 60
        assert img.num_channels == 5
        assert img.channel_names == ['Ag107', 'Pr141', 'Sm147', 'Eu153', 'Yb172']

    def test_read_tiff(self, analysis_cpout_images_path: Path):
        tiff_file_path = analysis_cpout_images_path / '20210305_NE_mockData1_s0_a1_ac_fullFiltered.tiff'
        img = MultichannelImage.read_tiff(tiff_file_path)
        assert img.width == 60
        assert img.height == 60
        assert img.num_channels == 5
        assert img.channel_names == ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5']

    def test_read_tiff_with_panel(self, analysis_cpout_images_path: Path):
        tiff_file_path = analysis_cpout_images_path / '20210305_NE_mockData1_s0_a1_ac_fullFiltered.tiff'
        panel = pd.DataFrame(data={'channel_name': ['Ag107', 'Pr141', 'Sm147', 'Eu153', 'Yb172']})
        img = MultichannelImage.read_tiff(tiff_file_path, panel=panel)
        assert img.width == 60
        assert img.height == 60
        assert img.num_channels == 5
        assert img.channel_names == ['Ag107', 'Pr141', 'Sm147', 'Eu153', 'Yb172']

    def test_read_ome_tiff(self, analysis_ometiff_path: Path):
        ome_tiff_file_path = analysis_ometiff_path / '20210305_NE_mockData1' / '20210305_NE_mockData1_s0_a1_ac.ome.tiff'
        img = MultichannelImage.read_tiff(ome_tiff_file_path)
        assert img.width == 60
        assert img.height == 60
        assert img.num_channels == 5
        assert img.channel_names == ['107Ag', 'Cytoker_651((3356))Pr141', 'Laminin_681((851))Sm147',
                                     'YBX1_2987((3532))Eu153', 'H3K27Ac_1977((2242))Yb172']

    def test_write_ome_tiff(self, analysis_ometiff_path: Path, tmp_path: Path):
        ome_tiff_file_path = analysis_ometiff_path / '20210305_NE_mockData1' / '20210305_NE_mockData1_s0_a1_ac.ome.tiff'
        img = MultichannelImage.read_tiff(ome_tiff_file_path)
        copy_file_path = tmp_path / '20210305_NE_mockData1_s0_a1_ac.ome.tiff'
        img.write_ome_tiff(copy_file_path)
        img = MultichannelImage.read_tiff(copy_file_path)
        assert img.width == 60
        assert img.height == 60
        assert img.num_channels == 5
        assert img.channel_names == ['107Ag', 'Cytoker_651((3356))Pr141', 'Laminin_681((851))Sm147',
                                     'YBX1_2987((3532))Eu153', 'H3K27Ac_1977((2242))Yb172']
