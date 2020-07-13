import dill
import os
import yaml

from Parameters import Param
from pathlib import Path
from Dataset import TIFFSeries, MultipageTIFF, ND2

def load_alldata(folder_path, load_results = False):

    folder_path = Path(folder_path)

    param = Param()
    with open(folder_path.joinpath('Parameters.yml')) as file:
        documents = yaml.full_load(file)
    for k in documents.keys():
        setattr(param, k, documents[k])

    res = None
    if load_results:
        res = dill.load(open(folder_path.joinpath('Results.pkl'), "rb"))

    if param.data_type == 'series':
        data = TIFFSeries(
            Path(param.expdir), param.morpho_name, 
            param.signal_name, data_type=param.data_type,
            step=param.step, bad_frames=param.bad_frames,
            max_time=param.max_time)

    elif param.data_type == 'multi':
        data = MultipageTIFF(
            Path(param.expdir), param.morpho_name,
            param.signal_name, data_type=param.data_type,
            step=param.step, bad_frames=param.bad_frames,
            max_time=param.max_time)

    elif param.data_type == 'nd2':
        data = ND2(
            Path(param.expdir), param.morpho_name,
            param.signal_name, data_type=param.data_type,
            step=param.step, bad_frames=param.bad_frames,
            max_time=param.max_time)

    return param, res, data