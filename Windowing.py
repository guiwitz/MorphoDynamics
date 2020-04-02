import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as plt
# from skimage.segmentation import find_boundaries
# from skimage.color import label2rgb
# from skimage.external.tifffile import imread, imsave
# from Segmentation import segment
from ArtifactGeneration import FigureHelper

plot = FigureHelper(not True)


def create_windows(c, I, J):
    """ Generate binary masks that represent the sampling windows. """

    # Compute the distance transform of the contour
    # (note that the contour must be represented as zeros in an array of ones)
    D, F = distance_transform_edt(-1==c, return_indices=True) # We perform both the distance transform and the so-called feature transform

    # Fill array with cumulative lengths of closest points on the contour
    L = np.zeros(c.shape)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            L[i, j] = c[F[0, i, j], F[1, i, j]]

    # Create sampling windows
    m = binary_fill_holes(-1 < c)  # Binary mask corresponding to the contour and its interior
    w = np.zeros((J, I) + c.shape, dtype=np.bool)
    b = np.linspace(0, np.amax(D * m), J + 1) # Radial coordinate
    for j in range(J):
        s = np.linspace(0, 1, int(I / 2 ** j) + 1) # "Curvilinear" coordinate
        for i in range(int(I / 2 ** j)):
            w[j, i] = m & (s[i] <= L) & (L < s[i+1]) & (b[j] <= D) & (D < b[j+1])

    # Artifact generation
    plot.imshow('Contour', c)
    plot.imshow('Mask', m.astype(np.int))
    plot.imshow('Distance transform', D)
    # plot.imshow('Contour length', l)
    plot.imshow('Sectors', L)
    # plot.imshow('Windows', windows)
    # plot.imshow('Labeled windows', label2rgb(windows))
    # plot.imshow('Window boundaries', boundaries.astype(np.uint8))
    plot.show()

    return w

def label_windows(windows):
    """ Create an image where the sampling windows are shown as regions with unique gray levels. """
    tiles = np.zeros(windows.shape[2:4], dtype=np.uint16)
    n = 1
    for j in range(windows.shape[0]):
        for i in range(windows.shape[1]):
            tiles[windows[j, i]] = n
            n += 1
    return tiles

def extract_signals(y, w):
    """ Extract the mean values of an image over the sampling windows. """
    signal = np.nan * np.ones(w.shape[0:2])
    for j in range(w.shape[0]):
        for i in range(w.shape[1]):
            if np.any(w[j, i]):
                signal[j, i] = np.mean(y[w[j, i]])
    return signal

def show_windows(w, b):
    """ Display the sampling-window boundaries and indices. """
    plt.imshow(b, cmap='gray', vmin=0, vmax=2)
    for j in range(w.shape[0]):
        for i in range(int(w.shape[1]/2**j)):
            if np.any(w[j, i]):
                p = center_of_mass(w[j, i]) # Window centers
                plt.text(p[1], p[0], str(i), color='yellow', fontsize=4, horizontalalignment='center', verticalalignment='center')
