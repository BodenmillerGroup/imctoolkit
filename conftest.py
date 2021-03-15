import pytest
import requests
import shutil

from imctoolkit import ImageSingleCellData, SpatialCellGraph
from pathlib import Path


def _download_and_extract_asset(tmp_dir_path: Path, asset_url: str):
    asset_file_path = tmp_dir_path / 'asset.tar.gz'
    response = requests.get(asset_url, stream=True)
    if response.status_code == 200:
        with asset_file_path.open(mode='wb') as f:
            f.write(response.raw.read())
    shutil.unpack_archive(asset_file_path, tmp_dir_path)


@pytest.fixture(scope='session')
def analysis_cpout_images_path(tmp_path_factory):
    tmp_dir_path: Path = tmp_path_factory.mktemp('analysis_cpout_images')
    _download_and_extract_asset(tmp_dir_path, 'https://github.com/BodenmillerGroup/TestData/releases/download/v1.0.0/210308_ImcTestData_analysis_cpout_images.tar.gz')
    yield tmp_dir_path / 'datasets' / '210308_ImcTestData' / 'analysis' / 'cpout' / 'images'
    shutil.rmtree(tmp_dir_path)


@pytest.fixture(scope='session')
def analysis_cpout_masks_path(tmp_path_factory):
    tmp_dir_path: Path = tmp_path_factory.mktemp('analysis_cpout_images')
    _download_and_extract_asset(tmp_dir_path, 'https://github.com/BodenmillerGroup/TestData/releases/download/v1.0.0/210308_ImcTestData_analysis_cpout_masks.tar.gz')
    yield tmp_dir_path / 'datasets' / '210308_ImcTestData' / 'analysis' / 'cpout' / 'masks'
    shutil.rmtree(tmp_dir_path)


@pytest.fixture(scope='session')
def analysis_ometiff_path(tmp_path_factory):
    tmp_dir_path: Path = tmp_path_factory.mktemp('analysis_ometiff')
    _download_and_extract_asset(tmp_dir_path, 'https://github.com/BodenmillerGroup/TestData/releases/download/v1.0.1/210308_ImcTestData_analysis_ometiff.tar.gz')
    yield tmp_dir_path / 'datasets' / '210308_ImcTestData' / 'analysis' / 'ometiff'
    shutil.rmtree(tmp_dir_path)


@pytest.fixture(scope='session')
def raw_path(tmp_path_factory):
    tmp_dir_path: Path = tmp_path_factory.mktemp('raw')
    _download_and_extract_asset(tmp_dir_path, 'https://github.com/BodenmillerGroup/TestData/releases/download/v1.0.0/210308_ImcTestData_raw.tar.gz')
    yield tmp_dir_path / 'datasets' / '210308_ImcTestData' / 'raw'
    shutil.rmtree(tmp_dir_path)


@pytest.fixture
def data(analysis_cpout_images_path: Path, analysis_cpout_masks_path: Path):
    img_file_path = analysis_cpout_images_path / '20210305_NE_mockData1_s0_a1_ac_fullFiltered.tiff'
    mask_file_path = analysis_cpout_masks_path / '20210305_NE_mockData1_s0_a1_ac_ilastik_s2_Probabilities_mask.tiff'
    channel_names = ['Ag107', 'Pr141', 'Sm147', 'Eu153', 'Yb172']
    return ImageSingleCellData(img_file_path, mask_file_path, channel_names=channel_names)


@pytest.fixture
def knn_graph(data: ImageSingleCellData):
    dist_mat = data.compute_cell_centroid_distances(metric='euclidean')
    return SpatialCellGraph.construct_knn_graph(data, dist_mat, k=5, cell_properties=True, cell_channel_properties=True)


@pytest.fixture
def dist_graph(data: ImageSingleCellData):
    dist_mat = data.compute_cell_centroid_distances(metric='euclidean')
    return SpatialCellGraph.construct_dist_graph(data, dist_mat, 15, cell_properties=True, cell_channel_properties=True)
