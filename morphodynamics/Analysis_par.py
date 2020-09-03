import numpy as np
from scipy.ndimage import center_of_mass
from tifffile import TiffWriter
from skimage.segmentation import find_boundaries
from scipy.interpolate import splev
from matplotlib.backends.backend_pdf import PdfPages
from .Settings import Struct
from .Segmentation import segment_threshold, extract_contour, segment_cellpose, track_cellpose, track_threshold, segment_farid
from .DisplacementEstimation import fit_spline, map_contours2, rasterize_curve, compute_length, compute_area, show_edge_scatter, align_curves, subdivide_curve, subdivide_curve_discrete, splevper, map_contours3
from .Windowing import create_windows, extract_signals, label_windows, show_windows
import matplotlib.pyplot as plt
from cellpose import models
import dask


def analyze_morphodynamics(data, param):
    # Figures and other artifacts
    if param.showSegmentation:
        tw_seg = TiffWriter(param.resultdir + 'Segmentation.tif')
    else:
        tw_seg = None
    if param.showWindows:
        pp = PdfPages(param.resultdir + 'Windows.pdf')
        tw_win = TiffWriter(param.resultdir + 'Windows.tif')
    else:
        pp = None
        tw_win = None

    if param.cellpose:
        model = models.Cellpose(model_type='cyto')
    else:
        model = None

    # Calibration of the windowing procedure
    location = param.location
    x = data.load_frame_morpho(0)
    if param.cellpose:
        m = segment_cellpose(model, x, param.diameter, location)
        m = track_cellpose(m, location)
    else:
        #m = segment_threshold(x, param.sigma, param.T(0) if callable(param.T) else param.T, location)
        m = segment_farid(x)
        m = track_threshold(m, location)

    # update location
    if location is None:
        location = center_of_mass(m)

    c = extract_contour(m)  # Discrete cell contour
    s = fit_spline(c, param.lambda_)
    c = rasterize_curve(x.shape, s, 0)
    w, J, I = create_windows(c, splev(0, s), depth=param.depth, width=param.width)
    Imax = np.max(I)

    # Structures that will be saved to disk
    res = Struct()
    res.I = I
    res.J = J
    res.spline = []
    res.param0 = []
    res.param = []
    res.displacement = np.zeros((I[0], data.K - 1))  # Projection of the displacement vectors
    res.mean = np.zeros((len(data.signalfile), J, Imax, data.K))  # Signals from the outer sampling windows
    res.var = np.zeros((len(data.signalfile), J, Imax, data.K))  # Signals from the outer sampling windows
    res.length = np.zeros((data.K,))
    res.area = np.zeros((data.K,))
    res.orig = np.zeros((data.K,))

    # Segment all images but don't select cell
    segmented = segment_all(data, param, model)
    if param.distributed == 'local' or param.distributed == 'cluster':
        segmented = dask.compute(segmented)[0]

    # do the tracking
    for k in range(0, data.K):

        m = segmented[k]

        # select cell to track in mask
        if param.cellpose:
            m = track_cellpose(m, location)
        else:
            m = track_threshold(m, location)

        location = 2*np.array(center_of_mass(m[::2,::2])) # Set the location for the next iteration. Use reduced image for speed

        # replace initial segmentation with aligned one
        segmented[k] = m

    # create contour, windows etc. in multiple parallelized loops
    s0prm_all = {k: None for k in range(0, data.K)}
    c_all = {k: None for k in range(0, data.K)}
    s_all = {k: None for k in range(0, data.K)}

    # extract the contour and fit a spline
    if param.distributed == 'local' or param.distributed == 'cluster':
        s_c = dask.compute([dask.delayed(contour_spline)(m, param) for m in segmented])[0]
    else:
        s_c = [contour_spline(m, param) for m in segmented]

    for k in range(0, data.K):
        s_all[k] = s_c[k][0]
        res.spline.append(s_c[k][0])

    # align curves across frames and rasterize the windows
    if param.distributed == 'local' or param.distributed == 'cluster':
        spline_out = dask.compute([dask.delayed(spline_align_rasterize)(s_all[k-1] if k > 0 else None, s_all[k], x.shape, k) for k in range(0, data.K)])[0]
    else:
        spline_out = [spline_align_rasterize(s_all[k-1] if k > 0 else None, s_all[k], x.shape, k) for k in range(0, data.K)]

    for k in range(0, data.K):
        s0prm_all[k] = spline_out[k][0]
        c_all[k] = spline_out[k][1]
        res.orig[k] = spline_out[k][2]
    res.orig = np.cumsum(res.orig)

    # map windows across frames
    if param.distributed == 'local' or param.distributed == 'cluster':
        output = dask.compute([dask.delayed(windowing_mapping)(
            c_all[k], c_all[k-1] if k > 0 else None,
            s_all[k], s_all[k-1] if k > 0 else None,
            res.orig, s0prm_all[k], J, I, k) for k in range(0, data.K)])[0]
    else:
        output = [windowing_mapping(
            c_all[k], c_all[k-1] if k > 0 else None,
            s_all[k], s_all[k-1] if k > 0 else None,
            res.orig, s0prm_all[k], J, I, k) for k in range(0, data.K)]

    # extract signals and calculate displacements
    for k in range(0, data.K):
        w = output[k]['w']
        t = output[k]['t']
        t0 = output[k]['t0']

        s = s_all[k]

        for ell in range(len(data.signalfile)):
            res.mean[ell, :, :, k], res.var[ell, :, :, k] = extract_signals(data.load_frame_signal(ell, k), w)  # Signals extracted from various imaging channels

        # Compute projection of displacement vectors onto normal of contour
        if 0 < k:
            s0 = s_all[k-1]
            u = np.asarray(splev(np.mod(t0, 1), s0, der=1))  # Get a vector that is tangent to the contour
            u = np.asarray([u[1], -u[0]]) / np.linalg.norm(u, axis=0)  # Derive an orthogonal vector with unit norm
            res.displacement[:, k - 1] = np.sum((np.asarray(splev(np.mod(t, 1), s)) - np.asarray(splev(np.mod(t0, 1), s0))) * u, axis=0)  # Compute scalar product with displacement vector

        # Save variables for archival
        res.spline.append(s)
        if 0 < k:
            res.param0.append(t0)
            res.param.append(t)

    return res


def windowing_mapping(c, c0, s, s0, origin, s0prm, J, I, k):

    output = {'w': None, 't': None, 't0': None}
    ori = origin[k]
    if k > 0:
        ori0 = origin[k-1]

    w = create_windows(c, splevper(ori, s), J, I) # Sampling windows

    t = None
    t0 = None
    if k > 0:
        p, t0 = subdivide_curve_discrete(c0, I[0], s0, splevper(ori0, s0))
        t = map_contours2(s0prm, s, t0, t0-ori0+ori)  # Parameters of the endpoints of the displacement vectors

    output['w'] = w
    output['t'] = t
    output['t0'] = t0

    return output


def segment_all(data, param, model):
    """Segment all frames and return a list of labelled masks. The correct
    label is not selected here"""

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


def contour_spline(m, param):

    c = extract_contour(m)  # Discrete cell contour
    s = fit_spline(c, param.lambda_)  # Smoothed spline curve following the contour
    return s, c


def spline_align_rasterize(s0, s, im_shape, k):

    origin = 0
    s0prm = None
    if k > 0:
        s0prm, origin = align_curves(s0, s, 0)
    c = rasterize_curve(im_shape, s, origin)  # Representation of the contour as a grayscale image

    return s0prm, c, origin
