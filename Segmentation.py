from skimage.external.tifffile import imread
import matplotlib.pyplot as plt
from skimage.exposure import histogram
from skimage.filters import gaussian
from skimage.measure import find_contours, label
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelmin
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np
from ArtifactGenerator import Plot

p = Plot(not True)

def segment(x, m0=None):
    if m0 == None:
        # Determine threshold for segmentation
        h, _ = histogram(x, source_range='dtype') # Compute histogram of image
        s = UnivariateSpline(range(0, 65536), h) # Interpolate the histogram counts using a smoothing spline
        n = np.arange(0, 65535, 1) # Create array with positions of histogram bins
        hs = s(n) # Evaluate smoothing spline
        n0 = np.argmax(hs) # Find position of maximum
        m = argrelmin(hs)[0] # Find positions of local minima
        m = m[hs[m] < 0.2*hs[n0]] # Remove local minima that are too strong
        m0 = m[n0 < m][0] # Select first local minimum after maximum

    p.imshow('Test image', x)
    if p.debug & (m0 == None):
        p.plotopen('Histogram')
        plt.plot(h, 'b', lw=0.1, zorder=50)
        # plt.xlim(0, 1000)
        plt.plot(n, hs, 'r', lw=0.1, zorder=100)
        plt.plot(n0, hs[n0], 'go', zorder=10)
        plt.plot(m0, hs[m0], 'yo', zorder=10)
        p.plotclose()
    p.show()

    # Segment image by thresholding
    y = gaussian(x, sigma=2, preserve_range=True) # Smooth input image with a Gaussian
    # y = x
    z = m0 < y # Threshold smoothed input

    # Keep only the largest region
    regions, nr = label(z, return_num=True) # Label each region with a unique integer
    sr = np.zeros((nr,)) # Allocate array of region sizes
    for k in range(nr):
        sr[k] = np.sum(regions == k+1) # Populate array
    k = np.argmax(sr) # Get index of largest region
    z = regions == k+1 # Create mask of largest region

    # Fill holes in mask
    z = binary_fill_holes(z)

    # Extract pixels along contour of region
    c = np.asarray(find_contours(z, 0, fully_connected='high')[0], dtype=np.int)

    # Artifact generation
    p.imshow('Segmented region', 255*(m0 < y).astype(np.uint8))
    p.imshow('Smoothed segmented region', 255*z.astype(np.uint8))
    p.imshow('Regions', regions)
    p.show()

    return c

def drawContourOld(shape, c):
    x = np.zeros(shape)
    n = 0
    for p in c:
        x[p[1], p[0]] += n+len(c)
        n += 1
    return x

# x = imread('C:\\Work\\UniBE 2\\Guillaume\\Example_Data\\FRET_sensors + actin\\Histamine\\Expt2\\w16TIRF-CFP\\RhoA_OP_his_02_w16TIRF-CFP_t53.tif')
# segment(x)