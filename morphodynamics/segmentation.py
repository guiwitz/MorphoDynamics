import matplotlib.pyplot as plt
from skimage.exposure import histogram
from skimage.filters import gaussian, threshold_otsu, farid
from skimage.morphology import binary_closing, binary_erosion, disk
from skimage.measure import find_contours, label, regionprops
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelmin
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import center_of_mass
import numpy as np
from tifffile import imread

from .splineutils import fit_spline

def segment_threshold(x, sigma, T):
    """Segment the cell image, possibly with automatic threshold selection."""

    # Determine the threshold based on the histogram, if not provided manually
    if T is None:
        h, _ = histogram(x, source_range="dtype")  # Compute histogram of image
        s = UnivariateSpline(
            range(0, 65536), h
        )  # Interpolate the histogram counts using a smoothing spline
        n = np.arange(0, 65535, 1)  # Create array with positions of histogram bins
        hs = s(n)  # Evaluate smoothing spline
        n0 = np.argmax(hs)  # Find position of maximum
        m = argrelmin(hs)[0]  # Find positions of local minima
        m = m[hs[m] < 0.2 * hs[n0]]  # Remove local minima that are too strong
        T = m[n0 < m][0]  # Select first local minimum after maximum

    # Segment image by thresholding
    if sigma > 0:
        y = gaussian(
            x, sigma=sigma, preserve_range=True
        )  # Smooth input image with a Gaussian
        # y = median_filter(x, 9)
    else:
        y = x
    z = T < y  # Threshold image
    regions = label(z)

    return regions


def segment_farid(x, threshold=1, minsize=500):
    """
    Segment an image using its gradient calculated using
    the Farid method.

    Parameters
    ----------
    x: 2d array
        image to segment
    threshold: float
        threshold on gradient image
    minsize: int
        remove binary objects smaller than this value

    Returns
    -------
    regions: 2d array
        labelled array

    """

    farid2 = farid(gaussian(x, 2, preserve_range=True)) > threshold

    farid3 = binary_closing(farid2, disk(3))
    farid4 = binary_erosion(farid3, disk(3))

    farid_lab = label(farid4)
    farid_reg = regionprops(farid_lab)
    farid_indices = np.array(
        [0] + [x.label if x.area > 500 else 0 for x in farid_reg]
    ).astype(int)
    regions = farid_indices[farid_lab] > 0
    regions = label(regions)

    return regions


def tracking(regions, location=None, seg_type="farid"):
    """
    Given a labelled mask, select one of the objects as cell.
    If a location is provided, pick closest object, otherwise
    pick largest.

    Parameters
    ----------
    regions: 2d array
        labeled mask of cells
    location: 1d array, optional
        position vector
    seg_type: str
        type of segmentation used, currently 'farid', 'cellpose', 'ilastik' or 'conv_paint'

    Returns
    -------
    sel_region: 2d array
        binary mask of cell

    """

    if seg_type == "ilastik":
        regions = label(regions == 1)
    # number of regions
    nr = np.max(regions)
    # if not location is given, keep largest regions
    # otherwise keep region closest to location
    if location is None:
        sr = np.zeros((nr,))
        for k in range(nr):
            if seg_type in ["farid", "ilastik", "conv_paint", "precomputed"]:
                sr[k] = np.sum(binary_fill_holes(regions == k + 1))
            elif seg_type == "cellpose":
                sr[k] = np.sum(regions == k + 1)
        k = np.argmax(sr)
        sel_region = binary_fill_holes(regions == k + 1)
    else:
        cm = np.zeros((nr, 2))
        for k in range(nr):
            cm[k] = center_of_mass(regions == k + 1)
        k = np.argmin([np.linalg.norm(cm0 - location) for cm0 in cm])
        if seg_type in ["farid", "ilastik", "conv_paint", "precomputed"]:
            sel_region = binary_fill_holes(regions == k + 1)
        elif seg_type == "cellpose":
            sel_region = regions == k + 1

    return sel_region


def segment_cellpose(model, x, diameter, location, flow_threshold=0.4, cellprob_threshold=0.0):
    """Segment image x using Cellpose. If model is None, a model is loaded"""

    if model is None:
        from cellpose import models
        model = models.Cellpose(model_type="cyto2")
    m, flows, styles, diams = model.eval(
        [x], diameter=diameter, channels=[[0, 0]],
        flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)
    m = m[0]
    return m

def segment_conv_paint(x, random_forest):
    """ Segment image x using a trained classifier and
    the conv paint module. """

    m = random_forest.segment_image_stack(x)
    m = m == 2
    regions = label(m)
    return regions


def extract_contour(mask):
    """ Extract pixels along contour of mask. """

    return np.asarray(find_contours(mask, 0, fully_connected="high")[0], dtype=int)


def contour_spline(m, smoothing):
    """
    Extract contour from binary image and fit a spline

    Parameters
    ----------
    m: 2d array
        binary image
    smoothing: float
        smoothing parameter as defined by splprep

    Returns
    -------
    s: spline tuple
    c: 2d array
        coordinates of contour

    """

    c = extract_contour(m)  # Discrete cell contour
    s, u = fit_spline(c, smoothing)  # Smoothed spline curve following the contour
    return s, u, c


def estimateBleaching(filename, K, shape):
    """Estimate the intensity decay due to bleaching."""

    x = np.zeros((K,) + shape, dtype=np.uint16)
    c = np.zeros((K,) + shape + (3,), dtype=np.uint8)
    I = np.zeros((K,))
    for k in range(K):
        x[k, :, :] = imread(filename + str(k + 1) + ".tif")  # Input image
    xmax = np.max(x)
    for k in range(K):
        c[k, :, :, 1] = 255.0 * x[k, :, :] / xmax
        m = threshold_otsu(x[k, :, :]) < x[k, :, :]
        c[k, :, :, 0] = 255 * m
        I[k] = np.mean(x[k][m])