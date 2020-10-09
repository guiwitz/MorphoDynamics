import numpy as np
from scipy.ndimage import center_of_mass
from scipy.interpolate import splev
from .segmentation import (
    segment_threshold,
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
from .windowing import create_windows, extract_signals
from .results import Results
import matplotlib.pyplot as plt
from cellpose import models
import dask


def analyze_morphodynamics(data, param):
    """
    Main function performing segmentation, windowing, signal
    extraction and displacement matching.

    Parameters
    ----------
    data: data object
        as returned by morphodynamics.dataset
    param: Param object
        As created by morphodyanmics.parameters.Param

    Returns
    -------
    res: Result object
        as created by morphodyanmics.results.Result

    """

    # check if dask is used for parallelization
    is_parallel = False
    if param.distributed == "local" or param.distributed == "cluster":
        is_parallel = True

    if param.cellpose:
        model = models.Cellpose(model_type="cyto")
    else:
        model = None

    location, J, I = calibration(data, param)

    # Result structures that will be saved to disk
    res = Results(
        J=J, I=I, num_time_points=data.K, num_channels=len(data.signalfile)
    )

    # Segment all images but don't select cell
    segmented = dask.compute(segment_all(data, param, model))[0]

    # do the tracking
    segmented = track_all(segmented, location, param)

    # get all splines
    s_all = spline_all(segmented, param.lambda_, is_parallel)

    # align curves across frames and rasterize the windows
    s0prm_all, c_all, ori_all = align_all(
        s_all, data.shape, param.n_curve, is_parallel
    )

    # origin shifts have been computed pair-wise. Calculate the cumulative
    # sum to get a "true" alignment on the first frame
    res.orig = np.array([ori_all[k] for k in range(data.K)])
    res.orig = np.cumsum(res.orig)

    # define windows for each frame and compute pairs of corresponding
    # points on successive splines for displacement measurement
    w_all, t_all, t0_all = window_map_all(
        s_all, s0prm_all, J, I, c_all, res.orig, param.n_curve, is_parallel
    )

    # Signals extracted from various imaging channels
    mean_signal, var_signal = extract_signal_all(data, w_all, J, I)

    # compute displacements
    res.displacement = compute_displacement(s_all, t_all, t0_all)

    # Save variables for archival
    res.spline = [s_all[k] for k in range(data.K)]
    res.param0 = [t0_all[k] for k in t0_all]
    res.param = [t_all[k] for k in t_all]
    res.mean = mean_signal
    res.var = var_signal

    return res


def calibration(data, param):
    """
    Segment the first frame to determine the initial location
    (if not provided) as well as the number of windows.

    Parameters
    ----------
    data: data object
        as returned by morphodynamics.dataset
    param: Param object
        As created by morphodyanmics.parameters.Param

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
    if param.cellpose:
        m = segment_cellpose(model, x, param.diameter, location)
        m = tracking(m, location, seg_type="cellpose")
    else:
        # m = segment_threshold(x, param.sigma, param.T(0) if callable(param.T) else param.T, location)
        m = segment_farid(x)
        m = tracking(m, location, seg_type="farid")

    # update location
    if location is None:
        location = center_of_mass(m)

    s, _ = contour_spline(m, param.lambda_)
    c = rasterize_curve(param.n_curve, m.shape, s, 0)
    _, J, I = create_windows(
        c, splev(0, s), depth=param.depth, width=param.width
    )

    return location, J, I


def spline_all(segmented, smoothing, is_parallel):
    """
    Convert a series of segmented binary masks into splines.

    Parameters
    ----------
    segmented: list of 2d arrays
        each array is a segmented binary image
    smoothing: float
        smoothing parameter used by splprep
    is_parallel: bool
        use dask or not

    Returns
    -------
    s_all: dict of spline objects
        each element k of the dictionary contains the spline of the
        corresponding frame k. The frame k-1 contains None

    """

    if is_parallel:
        fun_contour_spline = dask.delayed(contour_spline, nout=2)
    else:
        fun_contour_spline = contour_spline

    s_all = {k: None for k in range(-1, len(segmented))}
    for k in range(0, len(segmented)):
        s_all[k], _ = fun_contour_spline(segmented[k], smoothing)
    s_all = dask.compute(s_all)[0]
    return s_all


def align_all(s_all, im_shape, num_points, is_parallel):
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
    is_parallel: bool
        use dask or not

    Returns
    -------
    s0prm_all: dict of splines
        each element k is the spline of frame k xy-aligned
        on spline of frame k+1
    c_all: dict of 2d arrays
        each element is the rasterized contour
    ori_all: dict of floats
        each element k is the spline parameter shift to align
        spline of frame k+1 on spine of frame k

    """

    if is_parallel:
        fun_spline_align_rasterize = dask.delayed(
            spline_align_rasterize, nout=3
        )
    else:
        fun_spline_align_rasterize = spline_align_rasterize

    num_frames = len(s_all) - 1
    s0prm_all = {k: None for k in range(num_frames)}
    c_all = {k: None for k in range(-1, num_frames)}
    ori_all = {k: None for k in range(0, num_frames)}
    for k in range(num_frames):
        s0prm_all[k], c_all[k], ori_all[k] = fun_spline_align_rasterize(
            num_points, s_all[k - 1], s_all[k], im_shape, k > 0
        )
    s0prm_all, c_all, ori_all = dask.compute(s0prm_all, c_all, ori_all)

    return s0prm_all, c_all, ori_all


def window_map_all(
    s_all, s_shift_all, J, I, c_all, origins, num_points, is_parallel
):
    """
    Create windows for spline s and map its position to spline of
    previous frame s0 to measure displacements.

    Parameters
    ----------
    num_points: int
        number of points used for spline discretization
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
    c_all: dict of 2d arrays
        each element is the rasterized contour
    origins: list of floats
        each element k is the spline parameter shift to align
        spline of frame k+1 on spine of frame k
    is_parallel: bool
        use dask or not

    Returns
    -------
    w_all: dict of 3d list
        each element is a window-list as output by create_windows()
    t_all: dict of 1d array
        each element is a list of spline parameters defining closest locations
        on s_all[t] to points defined by t0_all on s_all[t-1]
    t0_all: 1d array
        each element is a list of spline paramters defining points centered on
        windows of frame corresponding to s_all[t-1]

    """

    if is_parallel:
        fun_windowing_mapping = dask.delayed(windowing_mapping, nout=3)
    else:
        fun_windowing_mapping = windowing_mapping

    num_frames = len(s_all) - 1
    w_all = {k: None for k in range(num_frames)}
    t_all = {k: None for k in range(num_frames)}
    t0_all = {k: None for k in range(num_frames)}
    # map windows accross frames
    for k in range(num_frames):
        w_all[k], t_all[k], t0_all[k] = fun_windowing_mapping(
            num_points,
            c_all[k],
            c_all[k - 1],
            s_all[k],
            s_all[k - 1],
            origins[k],
            origins[k - 1],
            s_shift_all[k],
            J,
            I,
            k > 0,
        )
    w_all, t_all, t0_all = dask.compute(w_all, t_all, t0_all)

    return w_all, t_all, t0_all


def extract_signal_all(data, w_all, J, I):
    """
    Extract signals from all frames.

    Parameters
    ----------
    data: data object
        as returned by morphodynamics.dataset
    w_all: dict of window-lists
        each element is a window index list as
        output by create_windows()
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

    mean_signal = np.zeros((len(data.signalfile), J, np.max(I), data.K))
    var_signal = np.zeros((len(data.signalfile), J, np.max(I), data.K))

    for k in range(data.K):
        for ell in range(len(data.signalfile)):
            (
                mean_signal[ell, :, :, k],
                var_signal[ell, :, :, k],
            ) = extract_signals(data.load_frame_signal(ell, k), w_all[k])

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

    for k in range(0, len(segmented)):

        m = segmented[k]

        # select cell to track in mask
        if param.cellpose:
            m = tracking(m, location, seg_type="cellpose")
        else:
            m = tracking(m, location, seg_type="farid")

        # Set the location for the next iteration. Use reduced image for speed
        location = 2 * np.array(center_of_mass(m[::2, ::2]))

        # replace initial segmentation with aligned one
        segmented[k] = m
    return segmented


def compute_displacement(s_all, t_all, t0_all):
    """
    Compute displacment between pairs of successive splines.

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
    num_time_points = len(s_all) - 1
    displacements = np.zeros((len(t_all[1]), num_time_points - 1))
    for k in range(num_time_points):
        s = s_all[k]
        # Compute projection of displacement vectors onto normal of contour
        if 0 < k:
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


def windowing_mapping(N, c, c0, s, s0, ori, ori0, s0_shifted, J, I, align):
    """
    Create windows for spline s and map its position to spline of
    previous frame s0 to measure displacements.

    Parameters
    ----------
    N: int
        number of points used for spline discretization
    c: 2d array
        rasterized image of spline s
    c0 : 2d array
        rasterized image of spline s0
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
    align: bool
        perform position matching or not

    Returns
    -------
    w: 3d list
        w[i][j][0] and w[0][0][1] are 1d arrays representing
        lists of x,y indices of pixels belonging to window in i'th layer
        in j'th window
    t: 1d array
        list of spline parameters defining closest locations
        on s to points defined by t0 on s0
    t0: 1d array
        list of spline paramters defining points centered on
        windows of frame corresponding to s0

    """

    w, _, _ = create_windows(c, splevper(ori, s), J, I)

    t = None
    t0 = None
    if align:
        p, t0 = subdivide_curve_discrete(N, c0, I[0], s0, splevper(ori0, s0))
        # Parameters of the endpoints of the displacement vectors
        t = map_contours2(s0_shifted, s, t0, t0 - ori0 + ori)

    return w, t, t0


def segment_all(data, param, model=None):
    """
    Segment all frames and return a list of labelled masks. The correct
    label is not selected here

    Parameters
    ----------
    data: data object
        as returned by morphodynamics.dataset
    param: Param object
        As created by morphodyanmics.parameters.Param
    model: cellpose model, optional

    Returns
    -------
    segmented: list of 2d arrays
        list of labelled arrays

    """

    # check if distributed computing should be used
    distr = False
    if param.distributed == "local" or param.distributed == "cluster":
        distr = True

    # Segment all images but don't do tracking (selection of label)
    segmented = []
    for k in range(0, data.K):
        if distr:
            x = dask.delayed(data.load_frame_morpho)(k)  # Input image
        else:
            x = data.load_frame_morpho(k)  # Input image

        if param.cellpose:
            if distr:
                m = dask.delayed(segment_cellpose)(
                    None, x, param.diameter, None
                )
            else:
                m = segment_cellpose(model, x, param.diameter, None)
        else:
            if distr:
                # m = dask.delayed(segment_threshold)(x, param.sigma, param.T(k) if callable(param.T) else param.T, None)
                m = dask.delayed(segment_farid)(x)
            else:
                # m = segment_threshold(x, param.sigma, param.T(k) if callable(param.T) else param.T, None)
                m = segment_farid(x)
        segmented.append(m)
    return segmented


def spline_align_rasterize(N, s0, s, im_shape, align):
    """
    Align a spline s with another spline s0 and provide a rasterized
    version of it.

    Parameters
    ----------
    N: int
        number of points used for spline discretization
    s0: tuple bspline object
        spline to align to
    s: tuple bspline object
        spline to align
    im_shape: tuple
        size of image for rasterization
    align: bool
        do alignement necessary or not

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

    return s0_shifted, c, origin