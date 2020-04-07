import numpy as np
from skimage.external.tifffile import TiffWriter  # , imread  # , imsave
from skimage.segmentation import find_boundaries
from scipy.interpolate import splev
from matplotlib.backends.backend_pdf import PdfPages
from Settings import Struct, load_settings
from FigureHelper import FigureHelper
from Segmentation import segment
from DisplacementEstimation import fit_spline, map_contours, rasterize_curve, compute_length, compute_area  # , show_edge_scatter
from Windowing import create_windows, extract_signals, label_windows, show_windows
# from SignalExtraction import showSignals


def analyze_morphodynamics(data, param):
    # Figures and other artifacts
    fh = FigureHelper(not True)
    if param.showWindows:
        pp = PdfPages(param.resultdir + 'Windows.pdf')

    # Structures that will be saved to disk
    res = Struct()
    res.spline = []
    res.param0 = []
    res.param = []
    res.displacement = np.zeros((param.I, data.K - 1))  # Projection of the displacement vectors
    res.signal = np.zeros((len(data.signalfile), param.J, param.I, data.K))  # Signals from the outer sampling windows
    res.length = np.zeros((data.K,))
    res.area = np.zeros((data.K,))

    from skimage.external.tifffile import imread
    mask = imread(param.resultdir + '../Mask.tif')

    # Main loop on frames
    deltat = 0
    tw = TiffWriter(param.resultdir + 'Segmentation.tif')
    for k in range(0, data.K):
        print(k)
        x = data.load_frame_morpho(k)  # Input image

        # c = segment(x, param.sigma, param.Tfun(k), tw)  # Discrete cell contour
        c = segment(x, param.sigma, param.T, tw, mask)  # Discrete cell contour
        s = fit_spline(c, param.lambda_)  # Smoothed spline curve following the contour
        res.length[k] = compute_length(s)  # Length of the contour
        res.area[k] = compute_area(s)  # Area delimited by the contour
        if 0 < k:
            t0 = deltat + 0.5 / param.I + np.linspace(0, 1, param.I, endpoint=False)  # Parameters of the startpoints of the displacement vectors
            t = map_contours(s0, s, t0)  # Parameters of the endpoints of the displacement vectors
            deltat += t[0] - t0[0]  # Translation of the origin of the spline curve
        c = rasterize_curve(x.shape, s, deltat)  # Representation of the contour as a grayscale image
        w = create_windows(c, param.I, param.J)  # Binary masks representing the sampling windows
        for m in range(len(data.signalfile)):
            res.signal[m, :, :, k] = extract_signals(data.load_frame_signal(m, k), w)  # Signals extracted from various imaging channels

        # Compute projection of displacement vectors onto normal of contour
        if 0 < k:
            u = np.asarray(splev(np.mod(t0, 1), s0, der=1))  # Get a vector that is tangent to the contour
            u = np.asarray([u[1], -u[0]]) / np.linalg.norm(u, axis=0)  # Derive an orthogonal vector with unit norm
            res.displacement[:, k - 1] = np.sum((np.asarray(splev(np.mod(t, 1), s)) - np.asarray(splev(np.mod(t0, 1), s0))) * u, axis=0)  # Compute scalar product with displacement vector

        # Artifact generation
        if param.showWindows:
            # if 0 < k:
            fh.open_figure('Frame ' + str(k), 1, (12, 9))
            show_windows(w, find_boundaries(label_windows(w)))  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
            # showEdge(s0, s, t0, t, res.displacement[:, k - 1])  # Show edge structures (spline curves, displacement vectors, sampling windows)
            pp.savefig()
            fh.show()
            # imsave(param.resultdir + 'Tiles.tif', 255 * np.asarray(w), compress=6)

        # Keep variable for the next iteration
        s0 = s

        # Save variables for archival
        res.spline.append(s)
        if 0 < k:
            res.param0.append(t0)
            res.param.append(t)

    # Close windows figure
    tw.close()
    if param.showWindows:
        pp.close()

    return res
