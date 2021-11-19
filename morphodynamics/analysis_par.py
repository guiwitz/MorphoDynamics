import os
import pickle
from pathlib import Path
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.interpolate import splev
import skimage.io
from .segmentation import (
    segment_cellpose,
    tracking,
    segment_farid,
    contour_spline,
)
from .displacementestimation import (
    map_contours2,
    rasterize_curve,
    align_curves,
    subdivide_curve_discrete,
    splevper,
)
from .windowing import create_windows, extract_signals, boundaries_image
from .results import Results
from .utils import load_alldata
from . import utils
import matplotlib.pyplot as plt
from tqdm import tqdm


def analyze_morphodynamics(
    data,
    param,
    client=None,
    only_seg=False,
    keep_seg=False,
    skip_segtrack=False,
):
    """
    Main function performing segmentation, windowing, signal
    extraction and displacement matching.

    Parameters
    ----------
    data: data object
        as returned by morphodynamics.dataset
    param: Param object
        As created by morphodyanmics.parameters.Param
    client: dask client
        client can be connected to either LocalCluster or SLURMCLuster
    only_seg: bool
        Perfrom only segmentation without windowing
    keep_seg: bool
        Store segmentation masks in memory
    skip_segtrack: bool
        Skip segmentation and tracking (only possible if done previously)

    Returns
    -------
    res: Result object
        as created by morphodyanmics.results.Result

    """

    if param.seg_algo == "cellpose":
        from cellpose import models
        model = models.Cellpose(model_type="cyto")
    else:
        model = None

    location, J, I = calibration(data, param, model)

    # Result structures that will be saved to disk
    res = Results(J=J, I=I, num_time_points=data.K, num_channels=len(data.signal_name))

    # create analysis folder if note existant
    analysis_path = param.analysis_folder.joinpath('segmented')
    if not os.path.isdir(analysis_path):
        os.makedirs(analysis_path)

    if not skip_segtrack:
        # Segment all images but don't select cell
        if param.seg_algo == "ilastik":
            segmented = np.arange(0, data.K)
        else:
            segmented = segment_all(data, param, client, model)

        if only_seg:
            return res

        # do the tracking
        segmented = track_all(segmented, location, param)

    # get all splines. s_all[k] is spline at frame k
    s_all = spline_all(data.K, param.lambda_, param, client)

    # align curves across frames and rasterize the windows
    s0prm_all, ori_all = align_all(s_all, data.shape, param.n_curve, param, client)

    # origin shifts have been computed pair-wise. Calculate the cumulative
    # sum to get a "true" alignment on the first frame
    res.orig = np.array([ori_all[k] for k in range(data.K)])
    res.orig = np.cumsum(res.orig)

    # create windows
    windowing_all(s_all, res.orig, param, J, I, client)

    # define windows for each frame and compute pairs of corresponding
    # points on successive splines for displacement measurement
    t_all, t0_all = window_map_all(
        s_all, s0prm_all, J, I, res.orig, param.n_curve, data.shape, param, client
    )

    # Signals extracted from various imaging channels
    mean_signal, var_signal = extract_signal_all(data, param, J, I)

    # compute displacements
    res.displacement = compute_displacement(s_all, t_all, t0_all)

    # Save variables for archival
    res.spline = [s_all[k] for k in range(data.K)]
    res.s0prm = [s0prm_all[k] for k in range(data.K)]
    res.param0 = [t0_all[k] for k in t0_all]
    res.param = [t_all[k] for k in t_all]
    res.mean = mean_signal
    res.var = var_signal

    return res


def calibration(data, param, model):
    """
    Segment the first frame to determine the initial location
    (if not provided) as well as the number of windows.

    Parameters
    ----------
    data: data object
        as returned by morphodynamics.dataset
    param: Param object
        As created by morphodyanmics.parameters.Param
    model: cellpose model, optional

    Returns
    -------
    location: tuple
        x,y location of cell
    J: int
        number of window layers
    I: list of int
        number of windows per layer

    """

    # Calibration of the windowing procedure
    location = param.location
    x = data.load_frame_morpho(0)
    if param.seg_algo == "cellpose":
        m = segment_cellpose(model, x, param.diameter, location)
        m = tracking(m, location, seg_type="cellpose")
    elif param.seg_algo == "ilastik":
        segpath = param.seg_folder
        num = str(0).zfill(
            len(next(segpath.glob("segmented_k_*.tif")).name.split("_")[-1]) - 4
        )
        m = skimage.io.imread(os.path.join(segpath, "segmented_k_" + num + ".tif"))
        m = tracking(m, location, seg_type="ilastik")
    elif param.seg_algo == "farid":
        # m = segment_threshold(x, param.sigma, param.T(0) if callable(param.T) else param.T, location)
        m = segment_farid(x)
        m = tracking(m, location, seg_type="farid")

    # update location
    if location is None:
        location = center_of_mass(m)

    I = [10, 0]
    J = 10
    try:
        s, _ = contour_spline(m, param.lambda_)
        c = rasterize_curve(param.n_curve, m.shape, s, 0)
        _, J, I = create_windows(c, splev(0, s), depth=param.depth, width=param.width)
    except Exception:
        print("I and J not calculated. Using 10 as default value.")

    return location, J, I


def import_and_spline(image_path, smoothing):
    """Import binary image at image_path and fit a spline to its contour.

    Parameters
    ----------
    image_path : path
        path to binary image
    smoothing : float
        smoothing factor for spline (see splrep)

    Returns
    -------
    s : spline object (tuple)
        spline fitted to binary object in image
    """

    m = skimage.io.imread(image_path)
    s, _ = contour_spline(m, smoothing)
    return s


def spline_all(num_frames, smoothing, param, client):
    """
    Convert a series of segmented binary masks into splines.

    Parameters
    ----------
    num_frames: int
        number of frames to analyze
    smoothing: float
        smoothing parameter used by splprep
    param: Param object
        As created by morphodyanmics.parameters.Param
    client: dask client
        client connected to LocalCluster or SLURMCLuster

    Returns
    -------
    s_all: dict of spline objects
        each element k of the dictionary contains the spline of the
        corresponding frame k. The frame k-1 contains None

    """

    save_path = os.path.join(param.analysis_folder, "segmented")

    s_all = {k: None for k in range(-1, num_frames)}
    for k in range(num_frames):
        name = os.path.join(save_path, "tracked_k_" + str(k) + ".tif")
        s_all[k] = client.submit(import_and_spline, name, smoothing)

    for k in tqdm(range(num_frames), "frames splining"):
        future = s_all[k]
        s_all[k] = future.result()
        future.cancel()
    return s_all


def align_all(s_all, im_shape, num_points, param, client):
    """
    Align all successive pairs of splines and generate for each spline
    a rasterized version.

    Parameters
    ----------
    s_all: dict of splines
        each element k is the spline of frame k
    im_shape: tuple
        intended shape of rasterized image
    num_points: int
        number of estimated spline points
    param : Param object
        As created by morphodyanmics.parameters.Param
    client: dask client
        client connected to LocalCluster or SLURMCluster

    Returns
    -------
    s0prm_all: dict of splines
        each element k is the spline of frame k xy-aligned
        on spline of frame k+1
    c_all: dict of 2d arrays
        each element is the rasterized contour
    ori_all: dict of floats
        each element k is the spline parameter shift to align
        spline of frame k on spline of frame k-1

    """

    save_path = os.path.join(param.analysis_folder, "segmented")

    num_frames = np.max(list(s_all.keys())) + 1
    s0prm_all = {k: None for k in range(-1, num_frames)}
    ori_all = {k: 0 for k in range(0, num_frames)}
    names = [
        os.path.join(save_path, "rasterized_k_" + str(k) + ".tif")
        for k in range(num_frames)
    ]
    spline_compute = {
        k: client.submit(
            spline_align_rasterize,
            num_points,
            s_all[k - 1],
            s_all[k],
            im_shape,
            k > 0,
            names[k],
        )
        for k in range(0, num_frames)
    }

    for k in tqdm(range(0, num_frames), "frames rasterize"):
        future = spline_compute[k]
        s0prm_all[k - 1], ori_all[k], _ = future.result()
        future.cancel()

    return s0prm_all, ori_all


def windowing(s, ori, param, J, I, k_iter):
    """Create windowing for frame k_iter and save results.

    Parameters
    ----------
    s : tuple
        spline object
    ori : float
        origin shift
    param : Param object
        As created by morphodyanmics.parameters.Param
    J : int
        number of window layers
    I : list of int
        number of windows per layer
    k_iter : int
        frame index

    """

    save_path = os.path.join(param.analysis_folder, "segmented")
    name = os.path.join(save_path, "window_k_" + str(k_iter) + ".pkl")
    name2 = os.path.join(save_path, "window_image_k_" + str(k_iter) + ".tif")

    c = utils.load_rasterized(save_path, k_iter)
    w, _, _ = create_windows(c, splevper(ori, s), J, I)

    pickle.dump(w, open(name, "wb"))
    b0 = boundaries_image(c.shape, w)
    skimage.io.imsave(name2, b0.astype(np.uint8), check_contrast=False)


def windowing_all(s_all, ori_all, param, J, I, client):
    """
    Create windowing for all splines in s_all.

    Parameters
    ----------
    s_all: dict of spline objects
        each element k of the dictionary contains the spline of the
        corresponding frame k. The frame k-1 contains None
    ori_all: list of floats
        each element k is the spline parameter shift to align
        spline of frame k+1 on spine of frame k
    param: Param object
        As created by morphodyanmics.parameters.Param
    J: int
        number of window layers
    I: list of int
        number of windows per layer
    client: dask client
        client connected to LocalCluster or SLURMCluster

    """
    max_index = np.max(list(s_all.keys())) + 1
    compute_windows = [
        client.submit(windowing, s_all[k], ori_all[k], param, J, I, k)
        for k in range(max_index)
    ]
    for k in tqdm(range(max_index), "frames compute windows"):
        compute_windows[k].result()
        compute_windows[k].cancel()


def window_map(N, s, s0, ori, ori0, s0_shifted, J, I, k_iter, param):
    """
    Create windows for spline s and map its position to spline of
    previous frame s0 to measure displacements.

    Parameters
    ----------
    N: int
        number of points used for spline discretization
    s: tuple bspline object
        spline to align
    s0: tuple bspline object
        spline to align to
    ori: float
        spline parameter shift to align s origin
    ori0: float
        spline parameter shift to align s0 origin
    s0_shifted: tuple bspline object
        shifted version of s0 xy-aligned on s
    J: int
        number of window layers
    I: list of int
        number of windows per layer
    k_iter: int
        frame index
    param: Param object
        As created by morphodyanmics.parameters.Param

    Returns
    -------
    t: 1d array
        list of spline parameters defining closest locations
        on s to points defined by t0 on s0
    t0: 1d array
        list of spline paramters defining points centered on
        windows of frame corresponding to s0

    """

    save_path = os.path.join(param.analysis_folder, "segmented")
    c0 = utils.load_rasterized(save_path, k_iter - 1)

    p, t0 = subdivide_curve_discrete(N, c0, I[0], s0, splevper(ori0, s0))
    # Parameters of the endpoints of the displacement vectors
    t = map_contours2(s0_shifted, s, t0, t0 - ori0 + ori)

    return t, t0


def window_map_all(
    s_all,
    s_shift_all,
    J,
    I,
    origins,
    num_points,
    im_shape,
    param,
    client,
):
    """
    Map all pairs of consecutive splines in s_all to minimize their
    "distance" as defined by a chosen functional. Return the set of
    optimized spline parameters.

    Parameters
    ----------
    s_all: dict of spline objects
        each element k of the dictionary contains the spline of the
        corresponding frame k. The frame k-1 contains None
    s_shift_all: dict of splines
        each element k is the spline of frame k xy-aligned
        on spline of frame k+1
    J: int
        number of window layers
    I: list of int
        number of windows per layer
    origins: list of floats
        each element k is the spline parameter shift to align
        spline of frame k+1 on spine of frame k
    num_points: int
        number of interpolation points
    param: Param object
        As created by morphodyanmics.parameters.Param
    client: dask client
        client connected to LocalCluster or SLURMCluster

    Returns
    -------
    t_all: dict of 1d array
        each element is a list of spline parameters defining closest locations
        on s_all[t] to points defined by t0_all on s_all[t-1]
    t0_all: 1d array
        each element is a list of spline paramters defining points centered on
        windows of frame corresponding to s_all[t-1]

    """

    num_frames = np.max(list(s_all.keys())) + 1
    # w_all = {k: None for k in range(num_frames)}
    t_all = {k: None for k in range(num_frames)}
    t0_all = {k: None for k in range(num_frames)}
    # map windows accross frames

    compute_window = {
        k: client.submit(
            window_map,
            num_points,
            s_all[k],
            s_all[k - 1],
            origins[k],
            origins[k - 1],
            s_shift_all[k - 1],
            J,
            I,
            k,
            param,
        )
        for k in range(1, num_frames)
    }
    for k in tqdm(range(1, num_frames), "frames compute mapping"):
        t_all[k], t0_all[k] = compute_window[k].result()
        compute_window[k].cancel()

    return t_all, t0_all


def extract_signal_all(data, param, J, I):
    """
    Extract signals from all frames.

    Parameters
    ----------
    data: data object
        as returned by morphodynamics.dataset
    param: Param object
        As created by morphodyanmics.parameters.Param
    J: int
        number of window layers
    I: list of int
        number of windows per layer

    Returns
    -------
    mean_signal: 4d array
        array with mean signal per window in all dimensions
    var_signal: 4d array
        array with variance of signal per window in all dimensions

    """

    save_path = os.path.join(param.analysis_folder, "segmented")

    mean_signal = np.zeros((len(data.signal_name), J, np.max(I), data.K))
    var_signal = np.zeros((len(data.signal_name), J, np.max(I), data.K))

    for k in range(data.K):
        name = os.path.join(save_path, "window_k_" + str(k) + ".pkl")
        w = pickle.load(open(name, "rb"))
        for ell in range(len(data.signal_name)):
            (
                mean_signal[ell, :, :, k],
                var_signal[ell, :, :, k],
            ) = extract_signals(data.load_frame_signal(ell, k), w)

    return mean_signal, var_signal


def track_all(segmented, location, param):
    """
    Turn the labelled arrays of segmented into binary images
    where one labelled object has been selected as cell

    Parameters
    ----------
    segmented: list of 2d arrays
        list of labelled images
    location: tuple
        x,y location to use for tracking across frames
    param: Param object
        As created by morphodyanmics.parameters.Param

    Returns
    -------
    segmented: list of 2d arrays
        list of binary images

    """

    # create analysis folder if note existant
    save_path = os.path.join(param.analysis_folder, "segmented")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for k in range(0, len(segmented)):

        segpath = param.seg_folder
        # m = segmented[k]
        num = k
        if param.seg_algo == "ilastik":
            num = str(k).zfill(
                len(next(segpath.glob("segmented_k_*.tif")).name.split("_")[-1]) - 4
            )
        m = skimage.io.imread(
            os.path.join(segpath, "segmented_k_" + str(num) + ".tif")
        )

        # select cell to track in mask
        m = tracking(m, location, seg_type=param.seg_algo)

        # Set the location for the next iteration. Use reduced image for speed
        location = 2 * np.array(center_of_mass(m[::2, ::2]))

        # replace initial segmentation with aligned one
        # segmented[k] = m
        m = m.astype(np.uint8)
        skimage.io.imsave(
            os.path.join(save_path, "tracked_k_" + str(k) + ".tif"),
            m,
            check_contrast=False,
        )

    return segmented


def compute_displacement(s_all, t_all, t0_all):
    """
    Compute displacement between pairs of successive splines.

    Parameters
    ----------
    s_all: dict of spline objects
        each element k of the dictionary contains the spline of the
        corresponding frame k. The frame k-1 contains None
    t_all: dict of 1d array
        each element is a list of spline parameters defining closest locations
        on s_all[t] to points defined by t0_all on s_all[t-1]
    t0_all: 1d array
        each element is a list of spline paramters defining points centered on
        windows of frame corresponding to s_all[t-1]

    """
    num_time_points = np.max(list(s_all.keys())) + 1
    displacements = np.zeros((len(t_all[1]), num_time_points - 1))
    for k in range(1, num_time_points):
        if t_all[k] is not None:
            s = s_all[k]
            # Compute projection of displacement vectors onto normal of contour
            s0 = s_all[k - 1]
            # Get a vector that is tangent to the contour
            u = np.asarray(splev(np.mod(t0_all[k], 1), s0, der=1))
            # Derive an orthogonal vector with unit norm
            u = np.asarray([u[1], -u[0]]) / np.linalg.norm(u, axis=0)
            # Compute scalar product with displacement vector
            displacements[:, k - 1] = np.sum(
                (
                    np.asarray(splev(np.mod(t_all[k], 1), s))
                    - np.asarray(splev(np.mod(t0_all[k], 1), s0))
                )
                * u,
                axis=0,
            )

    return displacements


def segment_single_frame(param, k, save_path):
    """Segment frame k of segmentation image defined in param object
    and save to disk.

    Parameters
    ----------
    param: Param object
        As created by morphodyanmics.parameters.Param
    k : int
        frame index
    save_path : path
        path to folder where to save segmented image
    """

    _, _, data = load_alldata(folder_path=None, load_results=False, param=param)
    x = data.load_frame_morpho(k)

    if param.seg_algo == "cellpose":
        m = segment_cellpose(None, x, param.diameter, None)
    elif param.seg_algo == "farid":
        m = segment_farid(x)
    elif param.seg_algo == "ilastik":
        filename = Path(save_path).joinpath("segmented_k_" + str(k) + ".tif")
        if filename.is_file():
            return None
        else:
            raise Exception(
                f"No segmentation file at {filename}. Run segmentation in ilastik first."
            )

    m = m.astype(np.uint8)

    m = skimage.io.imsave(
        os.path.join(save_path, "segmented_k_" + str(k) + ".tif"),
        m,
        check_contrast=False,
    )


def segment_all(data, param, client, model=None):
    """
    Segment all frames and return a list of labelled masks. The correct
    label is not selected here

    Parameters
    ----------
    data: data object
        as returned by morphodynamics.dataset
    param: Param object
        As created by morphodyanmics.parameters.Param
    client: dask client
        client connected to LocalCluster or SLURMCluster
    model: cellpose model, optional

    Returns
    -------
    segmented: list of 2d arrays
        list of labelled arrays

    """
    # create folder to store segmentation data
    save_path = param.seg_folder
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Segment all images but don't do tracking (selection of label)
    segmented = [
        client.submit(segment_single_frame, param, k, save_path)
        for k in range(0, data.K)
    ]
    for k in tqdm(range(0, data.K), "frame segmentation"):
        future = segmented[k]
        segmented[k] = future.result()
        future.cancel()
        del future

    return segmented


def spline_align_rasterize(N, s0, s, im_shape, align, filename):
    """
    Align a spline s with another spline s0 and provide a rasterized
    version of it.

    Parameters
    ----------
    N: int
        number of points used for spline discretization
    s0: tuple bspline object
        spline at time t-1
    s: tuple bspline object
        spline at time t
    im_shape: tuple
        size of image for rasterization
    align: bool
        do alignement necessary or not
    filename: path
        file name to save rasterized image

    Returns
    -------
    s0_shifted: tuple spline object
        shifted version of s0 matching s
    c: 2d array
        rasterized version of s
    origin: float
        spline parameter shift to align s origin on
        s0 origin

    """

    origin = 0
    s0_shifted = None
    if align:
        s0_shifted, origin = align_curves(N, s0, s, 0)
    c = rasterize_curve(N, im_shape, s, origin)
    skimage.io.imsave(filename, c, check_contrast=False)

    return s0_shifted, origin, c
