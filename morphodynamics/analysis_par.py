import numpy as np
from scipy.ndimage import center_of_mass
from scipy.interpolate import splev
from .segmentation import segment_threshold, segment_cellpose, tracking, segment_farid, contour_spline
from .displacementestimation import map_contours2, rasterize_curve, align_curves, subdivide_curve_discrete, splevper
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
    if param.distributed == 'local' or param.distributed == 'cluster':
        is_parallel = True

    # define dask functions if parallelized
    if is_parallel:
        fun_contour_spline = dask.delayed(contour_spline)
        fun_spline_align_rasterize = dask.delayed(spline_align_rasterize)
        fun_windowing_mapping = dask.delayed(windowing_mapping)
    else:
        fun_contour_spline = contour_spline
        fun_spline_align_rasterize = spline_align_rasterize
        fun_windowing_mapping = windowing_mapping

    if param.cellpose:
        model = models.Cellpose(model_type='cyto')
    else:
        model = None

    # Calibration of the windowing procedure
    location = param.location
    x = data.load_frame_morpho(0)
    if param.cellpose:
        m = segment_cellpose(model, x, param.diameter, location)
        m = tracking(m, location, seg_type='cellpose')
    else:
        #m = segment_threshold(x, param.sigma, param.T(0) if callable(param.T) else param.T, location)
        m = segment_farid(x)
        m = tracking(m, location, seg_type='farid')

    # update location
    if location is None:
        location = center_of_mass(m)

    s, _ = contour_spline(m, param.lambda_)
    c = rasterize_curve(param.n_curve, m.shape, s, 0)
    _, J, I = create_windows(c, splev(0, s), depth=param.depth, width=param.width)
    Imax = np.max(I)

    # Result structures that will be saved to disk
    res = Results(J=J, I=I, num_time_points=data.K, num_channels=len(data.signalfile))

    # Segment all images but don't select cell
    segmented = dask.compute(segment_all(data, param, model))[0]

    # do the tracking
    segmented = track_all(segmented, location, param)

    # initialize dictionaries for xy-shifted splines, 
    # rasterized contour, splines, windows, window-centered
    # spline paramters in pairs of successive points t and t0
    s0prm_all = {k: None for k in range(0, data.K)}
    c_all = {k: None for k in range(0, data.K)}
    s_all = {k: None for k in range(0, data.K)}
    w_all = {k: None for k in range(0, data.K)}
    t_all = {k: None for k in range(0, data.K)}
    t0_all = {k: None for k in range(0, data.K)}

    for k in range(0, data.K):
        s_all[k], _ = fun_contour_spline(segmented[k], param.lambda_)
        s_all[k] = dask.compute(s_all)[0]
    res.spline = [s_all[k] for k in s_all]

    # align curves across frames and rasterize the windows
    for k in range(0, data.K):
        s0prm_all[k], c_all[k], res.orig[k] = fun_spline_align_rasterize(
            param.n_curve, s_all[k-1] if k > 0 else None, s_all[k], x.shape, k > 0)
    s0prm_all, c_all, res.orig = dask.compute(s0prm_all, c_all, res.orig)

    # origin shifts have been computed pair-wise. Calculate the cumulative
    # sum to get a "true" alignment on the first frame
    res.orig = np.cumsum(res.orig)

    # map windows across frames
    for k in range(0, data.K):
        w_all[k], t_all[k], t0_all[k] = fun_windowing_mapping(
            param.n_curve,
            c_all[k], c_all[k-1] if k > 0 else None,
            s_all[k], s_all[k-1] if k > 0 else None,
            res.orig[k], res.orig[k-1] if k > 0 else None,
            s0prm_all[k], J, I, k > 0)
    w_all, t_all, t0_all = dask.compute(w_all, t_all, t0_all)

    # extract signals and calculate displacements
    for k in range(0, data.K):

        s = s_all[k]

        for ell in range(len(data.signalfile)):
            res.mean[ell, :, :, k], res.var[ell, :, :, k] = extract_signals(
                data.load_frame_signal(ell, k), w_all[k]
                )  # Signals extracted from various imaging channels

        # Compute projection of displacement vectors onto normal of contour
        if 0 < k:
            s0 = s_all[k-1]
            # Get a vector that is tangent to the contour
            u = np.asarray(splev(np.mod(t0_all[k], 1), s0, der=1))
            # Derive an orthogonal vector with unit norm
            u = np.asarray([u[1], -u[0]]) / np.linalg.norm(u, axis=0)
            res.displacement[:, k - 1] = np.sum(
                (
                    np.asarray(splev(np.mod(t_all[k], 1), s)) -
                    np.asarray(splev(np.mod(t0_all[k], 1), s0))
                ) * u, axis=0)  # Compute scalar product with displacement vector

        # Save variables for archival
    res.param0 = [t0_all[k] for x in t0_all]
    res.param = [t_all[k] for x in t_all]

    return res


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
            m = tracking(m, location, seg_type='cellpose')
        else:
            m = tracking(m, location, seg_type='farid')

        # Set the location for the next iteration. Use reduced image for speed
        location = 2*np.array(center_of_mass(m[::2, ::2]))

        # replace initial segmentation with aligned one
        segmented[k] = m
    return segmented


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
        t = map_contours2(s0_shifted, s, t0, t0-ori0+ori)  # Parameters of the endpoints of the displacement vectors

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
    if param.distributed == 'local' or param.distributed == 'cluster':
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
                m = dask.delayed(segment_cellpose)(None, x, param.diameter, None)
            else:
                m = segment_cellpose(model, x, param.diameter, None)
        else:
            if distr:
                #m = dask.delayed(segment_threshold)(x, param.sigma, param.T(k) if callable(param.T) else param.T, None)
                m = dask.delayed(segment_farid)(x)            
            else:
                #m = segment_threshold(x, param.sigma, param.T(k) if callable(param.T) else param.T, None)
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
