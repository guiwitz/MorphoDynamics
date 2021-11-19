import numpy as np

from morphodynamics import utils
from pathlib import Path
import create_dataset

analysis_folder = Path('synthetic/data/Results_ilastik')


def setup_module(module):

    create_dataset.make_dataset()


def count_files():

    tracked_files = analysis_folder.joinpath('segmented').glob('tracked*')
    assert len(list(tracked_files)) == 40

    raster_files = analysis_folder.joinpath('segmented').glob('raster*')
    assert len(list(raster_files)) == 40

    window_image_files = analysis_folder.joinpath('segmented').glob('window_image*')
    assert len(list(window_image_files)) == 40

    window_files = analysis_folder.joinpath('segmented').glob('window_k*')
    assert len(list(window_files)) == 40

    assert analysis_folder.joinpath('Results.pkl').is_file()
    assert analysis_folder.joinpath('Parameters.yml').is_file()


def test_load_rasterized():
    """Check that image is loaded"""

    image = utils.load_rasterized(analysis_folder.joinpath('segmented'), 1)
    assert type(image) is np.ndarray
    assert len(image.shape) == 2


def test_load_window_image():
    """Check that image is loaded"""

    image = utils.load_window_image(analysis_folder.joinpath('segmented'), 1)
    assert type(image) is np.ndarray
    assert len(image.shape) == 2
