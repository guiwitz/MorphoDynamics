import numpy as np
from skimage.external.tifffile import imread, imsave
from skimage.segmentation import find_boundaries
from scipy.interpolate import splev
import dill
from matplotlib.backends.backend_pdf import PdfPages

from Metadata import loadMetadata
from ArtifactGeneration import FigureHelper
from Segmentation import segment
from DisplacementEstimation import fitSpline, mapContours, showEdgeScatter, rasterizeCurve
from Windowing import createWindows, extractSignals, labelWindows, showWindows
from SignalExtraction import showSignals

# dataset = 'Synthetic'
# dataset = 'FRET_sensors + actinHistamineExpt2'
# dataset = 'FRET_sensors + actinPDGFRhoA_multipoint_0.5fn_s3_good'
# dataset = 'GBD_sensors + actinExpt_01'
dataset = 'Phantom'
path, morphosrc, sigsrc, K, T = loadMetadata(dataset)
# K = 10

# Analysis parameters
k0 = 0 # Index of first frame to be analyzed
I = 48 # Number of sampling windows in the outer layer (along the curve)
J = 5 # Number of sampling windows in the "radial" direction

# Figures and other artifacts
fh = FigureHelper(not True)
pp = PdfPages(fh.path + "Windows.pdf")

# Structures that will be saved to disk
spline = []
param0 = []
param = []
displacement = np.zeros((I, K - 1)) # Projection of the displacement vectors
signal = np.zeros((K, len(sigsrc), J, I)) # Signals from the outer sampling windows

# Main loop on frames
deltat = 0
for k in range(k0, K):
    print(k)
    x = imread(path + morphosrc + str(k + 1) + '.tif').astype(dtype=np.uint16) # Input image
    c = segment(x, T) # Discrete cell contour
    s = fitSpline(c) # Smoothed spline curve following the contour
    if k0 < k:
        t0 = deltat + 0.5 / I + np.linspace(0, 1, I, endpoint=False)  # Parameters of the startpoints of the displacement vectors
        t = mapContours(s0, s, t0) # Parameters of the endpoints of the displacement vectors
        deltat += t[0] - t0[0] # Translation of the origin of the spline curve
    c = rasterizeCurve(x.shape, s, deltat) # Representation of the contour as a grayscale image
    w = createWindows(c, I, J) # Binary masks representing the sampling windows
    for m in range(len(sigsrc)):
        signal[k, m] = extractSignals(imread(path + sigsrc[m](k + 1) + '.tif'), w) # Signals extracted from various imaging channels

    # Compute projection of displacement vectors onto normal of contour
    if k0 < k:
        u = np.asarray(splev(np.mod(t0, 1), s0, der=1)) # Get a vector that is tangent to the contour
        u = np.asarray([u[1], -u[0]]) / np.linalg.norm(u, axis=0) # Derive an orthogonal vector with unit norm
        displacement[:, k - k0 - 1] = np.sum((np.asarray(splev(np.mod(t, 1), s)) - np.asarray(splev(np.mod(t0, 1), s0))) * u, axis=0) # Compute scalar product with displacement vector

    # Artifact generation
    # if k0 < k:
    fh.openFigure('Frame ' + str(k), 1, (12, 9))
    showWindows(w, find_boundaries(labelWindows(w))) # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
    # showEdge(s0, s, t0, t, displacement[:, k - k0 - 1]) # Show edge structures (spline curves, displacement vectors, sampling windows)
    pp.savefig()
    fh.show()
    # imsave(plot.path + 'Tiles.tif', 255 * np.asarray(w), compress=6)

    # Keep variables for the next iteration
    s0 = s
    w0 = w

    # Save variables for archival
    spline.append(s)
    if k0 < k:
        param0.append(t0)
        param.append(t)

# Artifact generation
pp.close()
dic = {'path': path,
       'morphosrc': morphosrc,
       'sigsrc': sigsrc,
        'displacement': displacement,
        'signal': signal,
        'spline': spline,
        'param0': param0,
        'param': param}
dill.dump(dic, open(fh.path + 'Data.pkl', 'wb')) # Save analysis results to disk
showSignals()