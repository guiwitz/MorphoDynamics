import numpy as np
from scipy.ndimage import center_of_mass
from tifffile import TiffWriter
from skimage.segmentation import find_boundaries
from scipy.interpolate import splev
from matplotlib.backends.backend_pdf import PdfPages
from .settings import Struct
from .segmentation import segment_threshold, extract_contour, segment_cellpose, tracking, segment_farid
from .displacementestimation import fit_spline, map_contours2, rasterize_curve, compute_length, compute_area, show_edge_scatter, align_curves, subdivide_curve, subdivide_curve_discrete, splevper, map_contours3
from .windowing import create_windows, extract_signals, label_windows, show_windows
import matplotlib.pyplot as plt
from cellpose import models
import dask


def analyze_morphodynamics(data, param):
    # np.seterr(all='raise')

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
        m = tracking(m, location, seg_type='cellpose')
    else:
        m = segment_threshold(x, param.sigma, param.T(0) if callable(param.T) else param.T, location)
        #m = segment_farid(x)
        m = tracking(m, location, seg_type='farid')
    location = center_of_mass(m) #get location for tracking
    c = extract_contour(m)  # Discrete cell contour
    s = fit_spline(c, param.lambda_)
    c = rasterize_curve(param.n_curve, x.shape, s, 0)
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
        segmented = dask.delayed(segmented).compute()

    # do the tracking
    for k in range(0, data.K):
        print(k)

        m = segmented[k]

        # select cell to track in mask
        if param.cellpose:
            m = tracking(m, location, seg_type='cellpose')
        else:
            m = tracking(m, location, seg_type='farid')

        #if param.location is not None:
        location = center_of_mass(m) # Set the location for the next iteration
        c = extract_contour(m)  # Discrete cell contour

        s = fit_spline(c, param.lambda_)  # Smoothed spline curve following the contour

        if k > 0:
            s0prm, res.orig[k] = align_curves(param.n_curve, s0, s, res.orig[k-1]) # Intermediate curve and change of origin to account for cell motion

        c = rasterize_curve(param.n_curve, x.shape, s, res.orig[k])  # Representation of the contour as a grayscale image
        w, _, _ = create_windows(c, splevper(res.orig[k], s), J, I) # Sampling windows

        if k > 0:
            p, t0 = subdivide_curve_discrete(param.n_curve, c0, I[0], s0, splevper(res.orig[k-1], s0))
            t = map_contours2(s0prm, s, t0, t0-res.orig[k-1]+res.orig[k])  # Parameters of the endpoints of the displacement vectors

        for ell in range(len(data.signalfile)):
            res.mean[ell, :, :, k], res.var[ell, :, :, k] = extract_signals(data.load_frame_signal(ell, k), w)  # Signals extracted from various imaging channels

        # Compute projection of displacement vectors onto normal of contour
        if 0 < k:
            u = np.asarray(splev(np.mod(t0, 1), s0, der=1))  # Get a vector that is tangent to the contour
            u = np.asarray([u[1], -u[0]]) / np.linalg.norm(u, axis=0)  # Derive an orthogonal vector with unit norm
            res.displacement[:, k - 1] = np.sum((np.asarray(splev(np.mod(t, 1), s)) - np.asarray(splev(np.mod(t0, 1), s0))) * u, axis=0)  # Compute scalar product with displacement vector

        # Artifact generation
        if param.showSegmentation:
            tw_seg.save(255 * m.astype(np.uint8), compress=6)
        if param.showWindows:
            if 0 < k:
                plt.figure(figsize=(12, 9))
                plt.gca().set_title('Frame ' + str(k-1))
                plt.tight_layout()
                b0 = find_boundaries(label_windows(x.shape, w0))
                show_windows(w0, b0)  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
                # plt.plot(p1[:,1], p1[:,0], 'oy')
                # plt.plot(p1[0,1], p1[0,0], 'oc')
                show_edge_scatter(param.n_curve, s0, s, t0, t, res.displacement[:, k - 1])  # Show edge structures (spline curves, displacement vectors)
                # c00 = splev(np.linspace(0, 1, 10001), s0)
                # plt.plot(c00[0], c00[1], 'g')
                # cc = splev(np.linspace(0, 1, 10001), s)
                # plt.plot(cc[0], cc[1], 'b')
                # # c0prm = splev(np.linspace(0, 1, 10001), s0prm)
                # # plt.plot(c0prm[0], c0prm[1], 'r--')
                # # p0prm = splevper(t0, s0prm)
                # # plt.plot(p0prm[0], p0prm[1], 'or')
                # # plt.plot(p0prm[0][0], p0prm[1][0], 'oc')
                # # q0prm = splevper(t0-res.orig[k-1]+res.orig[k], s1)
                # # plt.plot(q0prm[0], q0prm[1], 'ob')
                # # plt.plot(q0prm[0][0], q0prm[1][0], 'oc')
                pp.savefig()
                tw_win.save(255 * b0.astype(np.uint8), compress=6)
                # plt.show()
                plt.close()

        # Keep variable for the next iteration
        s0 = s
        w0 = w
        c0 = c

        # Save variables for archival
        res.spline.append(s)
        if 0 < k:
            res.param0.append(t0)
            res.param.append(t)

    # Close windows figure
    if param.showSegmentation:
        tw_seg.close()
    if param.showWindows:
        pp.close()
        tw_win.close()

    return res


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
                m = dask.delayed(segment_threshold)(x, param.sigma, param.T(k) if callable(param.T) else param.T, None)
                #m = dask.delayed(segment_farid)(x)            
            else:
                m = segment_threshold(x, param.sigma, param.T(k) if callable(param.T) else param.T, None)
                #m = segment_farid(x)      
        segmented.append(m)
    return segmented
