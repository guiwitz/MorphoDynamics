import numpy as np

from morphodynamics import utils
from pathlib import Path

#exec(open("tests/create_dataset.py").read())

resultdir = Path('synthetic/data/Results_ilastik')


def count_files():

    tracked_files = resultdir.joinpath('segmented').glob('tracked*')
    assert len(list(tracked_files)) == 40

    raster_files = resultdir.joinpath('segmented').glob('raster*')
    assert len(list(raster_files)) == 40

    window_image_files = resultdir.joinpath('segmented').glob('window_image*')
    assert len(list(window_image_files)) == 40

    window_files = resultdir.joinpath('segmented').glob('window_k*')
    assert len(list(window_files)) == 40

    assert resultdir.joinpath('Results.pkl').is_file()
    assert resultdir.joinpath('Parameters.yml').is_file()


def test_load_rasterized():
    """Check that image is loaded"""

    image = utils.load_rasterized(resultdir.joinpath('segmented'), 1)
    assert type(image) is np.ndarray
    assert len(image.shape) == 2


def test_load_window_image():
    """Check that image is loaded"""

    image = utils.load_window_image(resultdir.joinpath('segmented'), 1)
    assert type(image) is np.ndarray
    assert len(image.shape) == 2
