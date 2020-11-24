import numpy as np

from morphodynamics import utils
from pathlib import Path

data_path = Path('synthetic/data/Results_ilastik/segmented')


def test_load_rasterized():
    """Check that image is loaded"""

    image = utils.load_rasterized(data_path, 1)
    assert type(image) is np.ndarray
    assert len(image.shape) == 2


def test_load_window_image():
    """Check that image is loaded"""

    image = utils.load_window_image(data_path, 1)
    assert type(image) is np.ndarray
    assert len(image.shape) == 2