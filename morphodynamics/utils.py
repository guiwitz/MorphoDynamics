import dill
import os
import yaml
import numpy as np
import skimage.io
import pandas as pd
from pathlib import Path
import zarr

from .store.parameters import Param
from .store.dataset import TIFFSeries, MultipageTIFF, ND2, H5, Nparray

# https://stackoverflow.com/a/2121918
import sys
from morphodynamics.store import results
sys.modules['morphodynamics.results'] = results


def load_alldata(folder_path, load_results=False, param=None):
    """
    Given a folder, load the parameter information contained in
    Parameters.yml and return the parameters, data and and optionally 
    the results loaded from the Results.yml file. If a Param object
    is given, it is not loaded again.

    Parameters
    ----------
    folder_path: str
        Path to folder containing Parameters.yml file
    load_results: bool (False)
        Load results from Results.yml file or not

    Returns
    -------
    param: Param object
        As created by morphodynamics.parameters.Param
    res: Result object
        As created by morphodynamics.results.Results
    data: Data object
        As created by morphodynamics.dataset

    """

    if param is not None:
        folder_path = Path(param.analysis_folder)
    else:
        folder_path = Path(folder_path)

        param = Param()
        with open(folder_path.joinpath("Parameters.yml")) as file:
            documents = yaml.full_load(file)
        for k in documents.keys():
            setattr(param, k, documents[k])
        param.analysis_folder = Path(param.analysis_folder)
        param.data_folder = Path(param.data_folder)
        param.seg_folder = Path(param.seg_folder)
        param.bad_frames_txt = param.bad_frames
        param.bad_frames = format_bad_frames(param.bad_frames)

    res = None
    if load_results:
        res = dill.load(open(folder_path.joinpath("Results.pkl"), "rb"))

    if param.data_type == "series":
        data = TIFFSeries(
            Path(param.data_folder),
            channel_name=[param.morpho_name]+param.signal_name,
            morpho_name=param.morpho_name,
            signal_name=param.signal_name,
            data_type=param.data_type,
            step=param.step,
            bad_frames=param.bad_frames,
            max_time=param.max_time,
        )

    elif param.data_type == "multi":
        data = MultipageTIFF(
            Path(param.data_folder),
            channel_name=[param.morpho_name]+param.signal_name,
            morpho_name=param.morpho_name,
            signal_name=param.signal_name,
            data_type=param.data_type,
            step=param.step,
            bad_frames=param.bad_frames,
            max_time=param.max_time,
        )

    elif param.data_type == "nd2":
        data = ND2(
            Path(param.data_folder),
            channel_name=[param.morpho_name]+param.signal_name,
            morpho_name=param.morpho_name,
            signal_name=param.signal_name,
            data_type=param.data_type,
            step=param.step,
            bad_frames=param.bad_frames,
            max_time=param.max_time,
        )

    elif param.data_type == "h5":
        data = H5(
            Path(param.data_folder),
            channel_name=[param.morpho_name]+param.signal_name,
            morpho_name=param.morpho_name,
            signal_name=param.signal_name,
            data_type=param.data_type,
            step=param.step,
            bad_frames=param.bad_frames,
            max_time=param.max_time,
        )

    elif param.data_type == "zarr":
        data = Nparray(
            nparray=zarr.open(param.data_folder),
            expdir=param.data_folder,
            morpho_name=param.morpho_name,
            signal_name=param.signal_name,
            data_type=param.data_type,
            step=param.step,
            bad_frames=param.bad_frames,
            max_time=param.max_time,
        )
    else:
        raise ValueError("Unknown data type")

    return param, res, data

def export_results_parameters(param, res):
    """Export parameters and results"""

    # export results
    dill.dump(
            res,
            open(os.path.join(param.analysis_folder, "Results.pkl"), "wb"),
        )

    # export parameters
    dict_file = {}
    for x in dir(param):
        if x[0] == "_":
            None
        elif (x == "analysis_folder") or (x == "data_folder") or (x == "seg_folder") or (x == "random_forest"):
            if getattr(param, x) is not None:
                dict_file[x] = Path(getattr(param, x)).as_posix()
            else:
                dict_file[x] = getattr(param, x)
        else:
            dict_file[x] = getattr(param, x)

    with open(param.analysis_folder.joinpath("Parameters.yml"), "w") as file:
        yaml.dump(dict_file, file)

    # export signals
    # export CSV data table
    signal_df = signalarray_to_dataframe({'mean': res.mean, 'var': res.var})
    signal_df.to_csv(os.path.join(param.analysis_folder, "Signals.csv"), index=False)


def dataset_from_param(param):
    """Given a param object, create the appropriate dataset."""
    
    if param.data_type == "zarr":
        data = zarr.open(param.data_folder)
        data = Nparray(
            nparray=data,
            expdir=param.data_folder,
            morpho_name=param.morpho_name,
            signal_name=param.signal_name,
            data_type=param.data_type,
            step=param.step,
            bad_frames=param.bad_frames,
            max_time=param.max_time,
        )

    elif param.data_type == "series":
            data = TIFFSeries(
                param.data_folder,
                morpho_name=param.morpho_name,
                signal_name=param.signal_name,
                data_type=param.data_type,
                step=param.step,
                bad_frames=param.bad_frames,
                max_time=param.max_time,
            )
    elif param.data_type == "multi":
        data = MultipageTIFF(
            param.data_folder,
            morpho_name=param.morpho_name,
            signal_name=param.signal_name,
            data_type=param.data_type,
            step=param.step,
            bad_frames=param.bad_frames,
            #switch_TZ=param.switch_TZ,
            max_time=param.max_time,
        )
    elif param.data_type == "nd2":
        data = ND2(
            param.data_folder,
            morpho_name=param.morpho_name,
            signal_name=param.signal_name,
            data_type=param.data_type,
            step=param.step,
            bad_frames=param.bad_frames,
            max_time=param.max_time,
        )
    elif param.data_type == "h5":
        data = H5(
            param.data_folder,
            morpho_name=param.morpho_name,
            signal_name=param.signal_name,
            data_type=param.data_type,
            step=param.step,
            bad_frames=param.bad_frames,
            max_time=param.max_time,
        )
    
    return data, param


def format_bad_frames(bad_frames):
    """Create an array of bad frame indices from string loaded from yml file."""

    if bad_frames == "":
        bads = []
    else:
        try:
            bads = [x.split("-") for x in bad_frames.split(",")]
            bads = [[int(x) for x in y] for y in bads]
            bads = np.concatenate(
                [
                    np.array(x) if len(x) == 1 else np.arange(x[0], x[1] + 1)
                    for x in bads
                ]
            )
        except:
            bads = []

    bads = list(bads)
    bads = [x.item() for x in bads]

    return bads


def load_rasterized(location, frame):
    """Load rasterized contour image at given frame"""

    image = skimage.io.imread(
        os.path.join(location, "rasterized_k_" + str(frame) + ".tif")
    )

    return image


def load_window_image(location, frame):
    """Load rasterized contour image at given frame"""

    image = skimage.io.imread(
        os.path.join(location, "window_image_k_" + str(frame) + ".tif")
    )

    return image


def signalarray_to_dataframe(signal_dict):
    """Turn a signal array with channel, layer, window, time dimensions
    into Dataframe"""

    dict_keys = list(signal_dict.keys())
    sarray = signal_dict[dict_keys[0]]
    # create array with indices
    timepoints = sarray.shape[3]
    windows = sarray.shape[2]
    layers = sarray.shape[1]
    colors = sarray.shape[0]
    all_indices = np.array([[[[[t, w, l, c] for t in range(timepoints)] for w in range(windows)] for l in range(layers)] for c in range(colors)])

    signal_df = pd.DataFrame(np.stack([
        np.ravel(all_indices[:,:,:,:,0]),
        np.ravel(all_indices[:,:,:,:,1]),
        np.ravel(all_indices[:,:,:,:,2]),
        np.ravel(all_indices[:,:,:,:,3])]).T, 
        columns=['time', 'window_index', 'layer_index', 'channel'])

    for k in dict_keys:
        signal_df[k] = np.ravel(signal_dict[k])

    return signal_df
