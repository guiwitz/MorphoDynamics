from skimage.external.tifffile import imread
import matplotlib.pyplot as plt
from skimage.exposure import histogram
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import find_contours, label
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelmin
from scipy.ndimage import median_filter
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np
from skimage.external.tifffile import imread, imsave
from ArtifactGeneration import FigureHelper

fh = FigureHelper(not True)

def segment_aux(x, sigma, T=None, tw=None):
    # Determine the threshold based on the histogram, if not provided manually
    if T is None:
        h, _ = histogram(x, source_range='dtype')  # Compute histogram of image
        s = UnivariateSpline(range(0, 65536), h)  # Interpolate the histogram counts using a smoothing spline
        n = np.arange(0, 65535, 1)  # Create array with positions of histogram bins
        hs = s(n)  # Evaluate smoothing spline
        n0 = np.argmax(hs)  # Find position of maximum
        m = argrelmin(hs)[0]  # Find positions of local minima
        m = m[hs[m] < 0.2*hs[n0]]  # Remove local minima that are too strong
        T = m[n0 < m][0]  # Select first local minimum after maximum

    # Artifact generation
    if fh.debug & (T is None):
        fh.open_figure('Histogram')
        plt.plot(h, 'b', lw=0.1, zorder=50)
        # plt.xlim(0, 1000)
        plt.plot(n, hs, 'r', lw=0.1, zorder=100)
        plt.plot(n0, hs[n0], 'go', zorder=10)
        plt.plot(T, hs[T], 'yo', zorder=10)
        fh.close_figure()
    fh.show()

    # Segment image by thresholding
    if sigma > 0:
        y = gaussian(x, sigma=sigma, preserve_range=True)  # Smooth input image with a Gaussian
        # y = median_filter(x, 9)
    else:
        y = x
    z = T < y  # Threshold image

    # Keep only the largest region
    regions, nr = label(z, return_num=True) # Label each region with a unique integer
    sr = np.zeros((nr,)) # Allocate array of region sizes
    for k in range(nr):
        sr[k] = np.sum(regions == k+1) # Populate array
    k = np.argmax(sr) # Get index of largest region
    z = regions == k+1 # Create mask of largest region

    # Fill holes in mask
    z = binary_fill_holes(z)
    # z[mask>0] = 0
    # if not (tw is None):
    #     tw.save(255 * z.astype(np.uint8), compress=6)

    return z


def segment(x, sigma, T=None, tw=None):  # , mask=None
    """ Segment the cell image, possibly with automatic threshold selection. """

    z = segment_aux(x, sigma, T, tw)

    # Extract pixels along contour of region
    c = np.asarray(find_contours(z, 0, fully_connected='high')[0], dtype=np.int)

    # Artifact generation
    fh.imshow('Input image', x)
    fh.imshow('Segmented image', 255 * (T < y).astype(np.uint8))
    fh.imshow('Filled largest segmented region', 255 * z.astype(np.uint8))
    # fh.imshow('All regions', regions)
    fh.show()

    return c  # , c.shape[0]/np.sum(z)


def estimateBleaching(filename, K, shape):
    """ Estimate the intensity decay due to bleaching. """
    x = np.zeros((K,) + shape, dtype=np.uint16)
    c = np.zeros((K,) + shape + (3,), dtype=np.uint8)
    I = np.zeros((K,))
    for k in range(K):
        x[k, :, :] = imread(filename + str(k + 1) + '.tif') # Input image
    xmax = np.max(x)
    for k in range(K):
        c[k, :, :, 1] = 255. * x[k, :, :] / xmax
        m = threshold_otsu(x[k, :, :]) < x[k, :, :]
        c[k, :, :, 0] = 255 * m
        I[k] = np.mean(x[k][m])
        # p.imshow('Segmentation', c[k, :, :, :])
        # p.show()
    imsave(fh.path + 'Segmentation.tif', c)
    fh.open_figure('Average intensity in segmented region')
    plt.plot(I)
    fh.close_figure()

# x = imread('C:\\Work\\UniBE 2\\Guillaume\\Example_Data\\FRET_sensors + actin\\Histamine\\Expt2\\w16TIRF-CFP\\RhoA_OP_his_02_w16TIRF-CFP_t53.tif')
# segment(x)

# calibrateBleaching('C:/Work/UniBE2/Guillaume/Example_Data/FRET_sensors + actin/Histamine/Expt2/w16TIRF-CFP/RhoA_OP_his_02_w16TIRF-CFP_t', 159, (358, 358))
# calibrateBleaching(r'C:\Work\UniBE2\Guillaume\Example_Data\FRET_sensors + actin\PDGF\RhoA_multipoint_0.5fn_s3_good\w34TIRF-mCherry\RhoA_multipoint_0.5fn_01_w34TIRF-mCherry_s3_t', 750, (358, 358))
