import numpy as np
from skimage.external.tifffile import imread  # , imsave
from skimage.segmentation import find_boundaries
from scipy.interpolate import splev
from matplotlib.backends.backend_pdf import PdfPages
from Metadata import Struct, load_data
from FigureHelper import FigureHelper
from Segmentation import segment
from DisplacementEstimation import fit_spline, map_contours, rasterize_curve, compute_length, compute_area  # , show_edge_scatter
from Windowing import create_windows, extract_signals, label_windows, show_windows
# from SignalExtraction import showSignals


def analyze_morphodynamics(ds, I, J, smooth_image, show_win):
    # ds.K = 3

    # Input and output directories
    datadir = 'C:/Work/UniBE2/Data/'
    resultdir = ds.name + '/'

    # Figures and other artifacts
    fh = FigureHelper(not True)
    if show_win:
        pp = PdfPages(resultdir + 'Windows.pdf')

    # Structures that will be saved to disk
    res = Struct()
    res.spline = []
    res.param0 = []
    res.param = []
    res.displacement = np.zeros((I, ds.K - 1))  # Projection of the displacement vectors
    res.signal = np.zeros((len(ds.signalfile), J, I, ds.K))  # Signals from the outer sampling windows
    res.length = np.zeros((ds.K,))
    res.area = np.zeros((ds.K,))

    # Main loop on frames
    deltat = 0
    for k in range(0, ds.K):
        print(k)
        x = ds.load_frame_morpho(k)  # Input image
        c = segment(x, ds.T, smooth_image)  # Discrete cell contour
        s = fit_spline(c)  # Smoothed spline curve following the contour
        res.length[k] = compute_length(s)  # Length of the contour
        res.area[k] = compute_area(s)  # Area delimited by the contour
        if 0 < k:
            t0 = deltat + 0.5 / I + np.linspace(0, 1, I, endpoint=False)  # Parameters of the startpoints of the displacement vectors
            t = map_contours(s0, s, t0)  # Parameters of the endpoints of the displacement vectors
            deltat += t[0] - t0[0]  # Translation of the origin of the spline curve
        c = rasterize_curve(x.shape, s, deltat)  # Representation of the contour as a grayscale image
        w = create_windows(c, I, J)  # Binary masks representing the sampling windows
        for m in range(len(ds.signalfile)):
            res.signal[m, :, :, k] = extract_signals(ds.load_frame_signal(m, k), w)  # Signals extracted from various imaging channels

        # Compute projection of displacement vectors onto normal of contour
        if 0 < k:
            u = np.asarray(splev(np.mod(t0, 1), s0, der=1))  # Get a vector that is tangent to the contour
            u = np.asarray([u[1], -u[0]]) / np.linalg.norm(u, axis=0)  # Derive an orthogonal vector with unit norm
            res.displacement[:, k - 1] = np.sum((np.asarray(splev(np.mod(t, 1), s)) - np.asarray(splev(np.mod(t0, 1), s0))) * u, axis=0)  # Compute scalar product with displacement vector

        # Artifact generation
        if show_win:
            # if 0 < k:
            fh.open_figure('Frame ' + str(k), 1, (12, 9))
            show_windows(w, find_boundaries(label_windows(w)))  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
            # showEdge(s0, s, t0, t, res.displacement[:, k - 1])  # Show edge structures (spline curves, displacement vectors, sampling windows)
            pp.savefig()
            fh.show()
            # imsave(resultdir + 'Tiles.tif', 255 * np.asarray(w), compress=6)

        # Keep variable for the next iteration
        s0 = s

        # Save variables for archival
        res.spline.append(s)
        if 0 < k:
            res.param0.append(t0)
            res.param.append(t)

    # Close windows figure
    if show_win:
        pp.close()

    return res
