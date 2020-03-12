import numpy as np
from skimage.external.tifffile import imread, imsave
from skimage.segmentation import find_boundaries
from scipy.interpolate import splev
import dill
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from Segmentation import segment
from DisplacementEstimation import fitSpline, mapContours, plotMap, rasterizeCurve
from Windowing import window, extractSignals, labelWindows
from ArtifactGenerator import Plot
from SignalExtraction import signalExtraction

plot = Plot(True)

# K = 50
# path = plot.path + 'Walking rectangles/'
# morphosrc = 'Phantom'
# sigsrc = []
# shape = (101, 101)
# T = None

K = 159
path = 'C:\\Work\\UniBE2\\Guillaume\\Example_Data\\FRET_sensors + actin\\Histamine\\Expt2\\'
morphosrc = 'w16TIRF-CFP\\RhoA_OP_his_02_w16TIRF-CFP_t'
sigsrc = [lambda k: 'ratio_tiffs\\ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(k, k, k),
          lambda k: 'w16TIRF-CFP\\RhoA_OP_his_02_w16TIRF-CFP_t' + str(k),
          lambda k: 'w26TIRFFRETacceptor\\RhoA_OP_his_02_w26TIRFFRETacceptor_t' + str(k),
          lambda k: 'w26TIRFFRETacceptor_corr\\RhoA_OP_his_02_w26TIRFFRETacceptor_t' + str(k),
          lambda k: 'w34TIRF-mCherry\\RhoA_OP_his_02_w34TIRF-mCherry_t' + str(k)]
shape = (358, 358)
T = None

# K = 750
# path = 'C:\\Work\\UniBE2\\Guillaume\\Example_Data\\FRET_sensors + actin\PDGF\\RhoA_multipoint_0.5fn_s3_good\\'
# morphosrc = 'w34TIRF-mCherry\\RhoA_multipoint_0.5fn_01_w34TIRF-mCherry_s3_t'
# sigsrc = [lambda k: 'ratio_tiffs\\photobleached_corrected_ratio_{:0>3d}'.format(k),
#           lambda k: 'w16TIRF-CFP\\RhoA_multipoint_0.5fn_01_w16TIRF-CFP_s3_t' + str(k),
#           lambda k: 'w26TIRFFRETacceptor\\RhoA_multipoint_0.5fn_01_w26TIRFFRETacceptor_s3_t' + str(k),
#           lambda k: 'w26TIRFFRETacceptor_corr\\RhoA_multipoint_0.5fn_01_w26TIRFFRETacceptor_s3_t' + str(k),
#           lambda k: 'w34TIRF-mCherry\\RhoA_multipoint_0.5fn_01_w34TIRF-mCherry_s3_t' + str(k)]
# shape = (358, 358)
# T = 2620

# K = 250
# path = 'C:\\Work\\UniBE2\\Guillaume\\Example_Data\\GBD_sensors + actin\\Expt_01\\'
# morphosrc = 'w14TIRF-GFP_s2\R52_LA-GFP_FN5_mCh-rGBD_02_w14TIRF-GFP_s2_t'
# sigsrc = [lambda k: 'w14TIRF-GFP_s2\R52_LA-GFP_FN5_mCh-rGBD_02_w14TIRF-GFP_s2_t' + str(k),
#           lambda k: 'w24TIRF-mCherry_s2\\R52_LA-GFP_FN5_mCh-rGBD_02_w24TIRF-mCherry_s2_t' + str(k)]
# shape = (716, 716)
# T = 165

pdf = PdfPages(plot.path + "Edge.pdf")

# Analysis parameters
kstart = 0 # Index of first frame to be analyzed
ncurv = 40 # Number of sampling windows along the curve
nrad = 5 # Number of "radial" sampling windows

# Array allocations
# wl = np.zeros((K,) + shape, dtype=np.uint16) # Labeled windows
# b = np.zeros((K,) + shape, dtype=np.uint8) # Boundaries of the sampling windows
displacement = np.zeros((ncurv, K - 1)) # Projection of the displacement vectors
signal = np.zeros((K, len(sigsrc), ncurv, nrad)) # Signals from the outer sampling windows

# Main loop on frames
deltat = 0
for k in range(kstart, K):
    print(k)
    x = imread(path + morphosrc + str(k + 1) + '.tif').astype(dtype=np.uint16) # Input image
    c = segment(x, T) # Discrete cell contour
    s = fitSpline(c) # Smoothed spline curve following the contour
    c = rasterizeCurve(x.shape, s) # Representation of the contour as a grayscale image
    # Positions of the displacement vectors along the splines
    if kstart < k:
        t0 = deltat + 0.5 / ncurv + np.linspace(0, 1, ncurv, endpoint=False)  # Parameters of the startpoints of the displacement vectors
        t = mapContours(s0, s, t0) # Parameters of the endpoints of the displacement vectors
        deltat += t[0] - t0[0]
        c[-1 < c] = np.mod(c[-1 < c] - deltat, 1)
    w = window(c, ncurv, nrad) # Binary masks for the sampling windows
    # imsave(plot.path + 'Tiles.tif', 255 * np.asarray(w), compress=6)
    signal[k] = extractSignals(path, sigsrc, k, w) # Signals extracted from various imaging channels

    # Compute projection of displacement vectors onto normal of contour
    if kstart < k:
        d1 = np.asarray(splev(np.mod(t0, 1), s0, der=1))
        d1n = np.linalg.norm(d1, axis=0)
        d1 = np.asarray([d1[1] / d1n, -d1[0] / d1n])
        displacement[:, k - kstart - 1] = np.sum((np.asarray(splev(np.mod(t, 1), s)) - np.asarray(splev(np.mod(t0, 1), s0))) * d1, axis=0)

    # Plot edge structures (spline curves, displacement vectors, sampling windows)
    if kstart < k:
        plot.plotopen('Frame ' + str(k), 1)
        plotMap(find_boundaries(labelWindows(w0)), w0, s0, s, t0, t, displacement[:, k-kstart-1], d1) # w0[0, 0].astype(dtype=np.uint8)
        plot.plotclose(False)
        pdf.savefig(plt.gcf())
        # plot.show()

    # Keep variables for the next iteration
    s0 = s
    x0 = x
    w0 = w

pdf.close()

# Save analysis results to disk
with open(plot.path + 'Signals.pkl', 'wb') as f:
    # pickle.dump({'morphosrc': morphosrc, 'sigsrc': sigsrc, 'displacement': displacement, 'signal': signal}, f)
    dill.dump({'morphosrc': morphosrc, 'sigsrc': sigsrc, 'displacement': displacement, 'signal': signal}, f)

# Plot results
signalExtraction()