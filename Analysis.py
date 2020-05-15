import numpy as np
from skimage.external.tifffile import TiffWriter
from skimage.segmentation import find_boundaries
from scipy.interpolate import splev
from matplotlib.backends.backend_pdf import PdfPages
from Settings import Struct
from Segmentation import segment
from DisplacementEstimation import fit_spline, map_contours2, rasterize_curve, compute_length, compute_area, \
    show_edge_scatter, align_curves, subdivide_curve
from Windowing import create_windows, extract_signals, label_windows, show_windows
import matplotlib.pyplot as plt


def analyze_morphodynamics(data, param):
    # Figures and other artifacts
    if param.showWindows:
        pp = PdfPages(param.resultdir + 'Windows.pdf')
        tw_win = TiffWriter(param.resultdir + 'Windows.tif')
    if param.showSegmentation:
        tw_seg = TiffWriter(param.resultdir + 'Segmentation.tif')
    else:
        tw_seg = None

    # Calibration of the windowing procedure
    x = data.load_frame_morpho(0)
    c = segment(x, param.sigma, param.Tfun(0) if hasattr(param, 'Tfun') else param.T)
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

    # Main loop on frames
    for k in range(0, data.K):
        print(k)
        x = data.load_frame_morpho(k)  # Input image

        c = segment(x, param.sigma, param.Tfun(k) if hasattr(param, 'Tfun') else param.T, tw_seg)  # Discrete cell contour
        s = fit_spline(c, param.lambda_)  # Smoothed spline curve following the contour
        res.length[k] = compute_length(s)  # Length of the contour
        res.area[k] = compute_area(s)  # Area delimited by the contour
        if 0 < k:
            s0prm, res.orig[k] = align_curves(s0, s, res.orig[k-1])
            # TODO: align the origins of the displacement vectors with the windows
            # t0 = res.orig[k-1] + 0.5 / I[0] + np.linspace(0, 1, I[0], endpoint=False)  # Parameters of the startpoints of the displacement vectors
            # t = res.orig[k] + 0.5 / I[0] + np.linspace(0, 1, I[0], endpoint=False)  # Parameters of the startpoints of the displacement vectors
            t0 = subdivide_curve(s0, res.orig[k-1], I[0])
            t = subdivide_curve(s, res.orig[k], I[0])
            t = map_contours2(s0prm, s, t0, t)  # Parameters of the endpoints of the displacement vectors
        c = rasterize_curve(x.shape, s, res.orig[k])  # Representation of the contour as a grayscale image
        w = create_windows(c, splev(res.orig[k], s), J, I, param.depth, param.width) # Sampling windows
        for m in range(len(data.signalfile)):
            res.mean[m, :, :, k], res.var[m, :, :, k] = extract_signals(data.load_frame_signal(m, k), w)  # Signals extracted from various imaging channels

        # Compute projection of displacement vectors onto normal of contour
        if 0 < k:
            u = np.asarray(splev(np.mod(t0, 1), s0, der=1))  # Get a vector that is tangent to the contour
            u = np.asarray([u[1], -u[0]]) / np.linalg.norm(u, axis=0)  # Derive an orthogonal vector with unit norm
            res.displacement[:, k - 1] = np.sum((np.asarray(splev(np.mod(t, 1), s)) - np.asarray(splev(np.mod(t0, 1), s0))) * u, axis=0)  # Compute scalar product with displacement vector

        # Artifact generation
        if param.showWindows:
            if 0 < k:
                plt.figure(figsize=(12, 9))
                plt.gca().set_title('Frame ' + str(k-1))
                plt.tight_layout()
                b0 = find_boundaries(label_windows(x.shape, w0))
                show_windows(w0, b0)  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
                show_edge_scatter(s0, s, t0, t, res.displacement[:, k - 1])  # Show edge structures (spline curves, displacement vectors)
                pp.savefig()
                tw_win.save(255 * b0.astype(np.uint8), compress=6)
                plt.close()

        # Keep variable for the next iteration
        s0 = s
        w0 = w

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
