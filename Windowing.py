import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import find_boundaries
from skimage.color import label2rgb
from skimage.external.tifffile import imread, imsave
from Segmentation import segment
from ArtifactGenerator import Plot

plot = Plot(not True)

def window(c, ncurv, nrad):
    mask = binary_fill_holes(-1 < c)  # Binary mask including the inside of the contour

    # Compute the distance transform of the contour
    # (note that the contour must be represented as zeros in an array of ones)
    # d = np.ones(c.shape) #
    # for p in c:
    #     d[p[1], p[0]] = 0
    D, F = distance_transform_edt(-1==c, return_indices=True) # We perform both the distance transform and the so-called feature transform

    # # Construct an image that stores the cumulative length of the contour, along the contour
    # l = np.zeros(c.shape)
    # q = c[0]
    # L = 0
    # for p in c:
    #     L += np.linalg.norm(p-q)
    #     l[p[1], p[0]] = L
    #     q = p

    # Fill array with cumulative lengths of closest points on the contour
    circ = np.zeros(c.shape)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            circ[i, j] = c[F[0, i, j], F[1, i, j]]

    # Create sampling windows
    s = np.linspace(0, 1, ncurv+1) # "Curvilinear" coordinate
    b = np.linspace(0, np.amax(D*mask), nrad+1) # Radial coordinate
    windows = np.zeros((ncurv, nrad, c.shape[0], c.shape[1]), dtype=np.bool)
    for i in range(ncurv):
        for j in range(nrad):
            windows[i, j] = mask & (s[i] <= circ) & (circ < s[i+1]) & (b[j] <= D) & (D < b[j+1])

    # Artifact generation
    plot.imshow('Contour', c)
    plot.imshow('Mask', mask.astype(np.int))
    plot.imshow('Distance transform', D)
    # plot.imshow('Contour length', l)
    plot.imshow('Sectors', circ)
    # plot.imshow('Windows', windows)
    # plot.imshow('Labeled windows', label2rgb(windows))
    # plot.imshow('Window boundaries', boundaries.astype(np.uint8))
    plot.show()

    return windows

def labelWindows(windows):
    tiles = np.zeros([windows.shape[2], windows.shape[3]], dtype=np.uint16)
    for i in range(windows.shape[0]):
        # for j in range(windows.shape[1]):
        for j in range(windows.shape[1]):
            tiles[windows[i, j]] = 1 + windows.shape[1]*i + j
    return tiles

def extractSignals(path, sigsrc, k, w):
    # Extract signals
    M = len(sigsrc)
    signal = np.zeros((M, w.shape[0], w.shape[1]))
    for m in range(M):
        y = imread(path + sigsrc[m](k+1) + '.tif')
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                if np.any(w[i, j]):
                    signal[m, i, j] = np.mean(y[w[i, j]])
                else:
                    signal[m, i, j] = np.nan
    return signal