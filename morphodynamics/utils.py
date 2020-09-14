import dill
import os
import yaml
import numpy as np
from pathlib import Path

from .parameters import Param
from .dataset import TIFFSeries, MultipageTIFF, ND2


def load_alldata(folder_path, load_results = False):

    folder_path = Path(folder_path)

    param = Param()
    with open(folder_path.joinpath('Parameters.yml')) as file:
        documents = yaml.full_load(file)
    for k in documents.keys():
        setattr(param, k, documents[k])
    param.bad_frames_txt = param.bad_frames
    param.bad_frames = format_bad_frames(param.bad_frames)

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

def format_bad_frames(bad_frames):
    if bad_frames == '' :
        bads = []
    else:
        try:
            bads = [x.split('-') for x in bad_frames.split(',')]
            bads = [[int(x) for x in y] for y in bads]
            bads = np.concatenate([np.array(x) if len(x)==1 else np.arange(x[0],x[1]+1) for x in bads])
        except:
            bads = []

    bads = list(bads)
    bads = [x.item() for x in bads]

    return bads
