import pytest
import numpy as np
from morphodynamics import analysis_par
from morphodynamics.store.dataset import H5
import os
from dask.distributed import Client, LocalCluster

from morphodynamics.store.parameters import Param
from morphodynamics.store.results import Results


@pytest.fixture
def demo_data():
    def _demo_data(output_folder):
    
        data_folder = 'synthetic/data'
        segmentation_folder = "synthetic/data/Ilastiksegmentation"
        analysis_folder = output_folder
        morpho_name = 'synth_ch1.h5'
        signal_name = ['synth_ch2.h5','synth_ch3.h5']

        param = Param(
            data_folder=data_folder,analysis_folder=analysis_folder,
            seg_folder=segmentation_folder, data_type="H5", morpho_name=morpho_name,
            signal_name=signal_name, seg_algo="ilastik")
        param.max_time = 10

        data = H5(
            expdir=data_folder,
            signal_name=signal_name,
            morpho_name=morpho_name,
            max_time=param.max_time,
        )
        
        return data, param
    return _demo_data

def test_data(demo_data):
    data, param = demo_data(("morphodynamics/tests/output/Results_step"))
    assert data.signal_name == ['synth_ch2.h5','synth_ch3.h5']
    assert data.morpho_name == 'synth_ch1.h5'
    assert param.seg_algo == 'ilastik'


def test_analyze_morphodynamics(demo_data):
    data, param = demo_data(("morphodynamics/tests/output/Results_step"))
    res = analysis_par.analyze_morphodynamics(
        data=data, param=param, client=None)
    assert isinstance(res, Results), "Result is not a Results object"
    assert param.analysis_folder.joinpath("segmented","tracked_k_9.tif").exists(), "Track image 9 missing"
    assert param.analysis_folder.joinpath("segmented","window_image_k_9.tif").exists(), "Window image 9 missing"
    assert param.analysis_folder.joinpath("segmented","window_k_9.pkl").exists(), "Window pkl data 9 missing"

def test_analyze_morphodynamics_dask(demo_data):
    data, param = demo_data(("morphodynamics/tests/output/Results_step_dask"))
    with Client() as client:
        res = analysis_par.analyze_morphodynamics(
            data=data, param=param, client=client)
        assert isinstance(res, Results), "Result is not a Results object"
        assert param.analysis_folder.joinpath("segmented","tracked_k_9.tif").exists(), "Track image 9 missing"
        assert param.analysis_folder.joinpath("segmented","window_image_k_9.tif").exists(), "Window image 9 missing"
        assert param.analysis_folder.joinpath("segmented","window_k_9.pkl").exists(), "Window pkl data 9 missing"

