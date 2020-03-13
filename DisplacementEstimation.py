from skimage.external.tifffile import imread, imsave
# from skimage.exposure import histogram
# from skimage.filters import gaussian
# from skimage.segmentation import find_boundaries
# from skimage.measure import find_contours
# from scipy.interpolate import UnivariateSpline
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass
# from scipy.signal import argrelmin
from scipy.interpolate import splprep, splev
from scipy.optimize import least_squares
import numpy as np
# import math
from numpy.linalg import norm
from skimage.segmentation import find_boundaries

from Segmentation import segment
from FunctionalDefinition import Functional
from ArtifactGenerator import Plot
from Windowing import labelWindows

plot = Plot(not True)

def fitSpline(c):
    """" Fit spline to contour specified as list of pixels. """
    # s = 0
    s = 1e2
    # s = 1e3
    # tck, u = splprep([[0, 50, 50, 0], [0, 0, 100, 100]], s=0, per=4)
    tck, u = splprep([c[:, 1], c[:, 0]], s=s, per=c.shape[0])
    return tck

def correlate(x, y):
    """ Compute the correlation between two signals with periodic boundary conditions. """
    z = np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(y[::-1])))
    return z

def mapContours(s1, s2, t1):
    """ Compute displacement vectors between two consecutive contours. """
    # Positions of the velocity arrows
    N = len(t1)
    t2 = t1

    # Weight for the cost function
    # w = 0
    w = np.sum((np.concatenate(splev(t2, s2)) - np.concatenate(splev(t1, s1))) ** 2) / (N - 1)
    # w = np.sum(1 / (np.concatenate(splev(t2, tck2)) - np.concatenate(splev(t1, s1)))**2) * (N-1)
    # w = 1e6

    # Lower and upper bounds for the least-squares problem
    lb = np.zeros((N+1,))
    lb[0] = -np.inf
    lb[N] = -1
    ub = np.inf * np.ones((N+1,))

    # Solve least-squares problem
    functional = Functional(s1, s2, t1, w)
    result = least_squares(functional.f, functional.transform(t2), bounds=(lb,ub), ftol=1e-3)
    t2 = functional.inversetransform(result.x)

    return t2

def showEdge(k, x, w, s1, s2, t1, t2, d, d1, ncurv, pdf):
    # Evaluate splines at various points
    c1 = splev(np.mod(t1, 1), s1)
    c2 = splev(np.mod(t2, 1), s2)
    c1p = splev(np.linspace(0, 1, 10001), s1)
    c2p = splev(np.linspace(0, 1, 10001), s2)

    # Calculate window centers
    p = np.zeros((w.shape[1], 2))
    for i in range(w.shape[1]):
        p[i] = center_of_mass(w[0, i])

    # Interpolate displacements
    # d = 0.5 + 0.5 * d / np.max(np.abs(d))
    d = np.interp(np.linspace(0, 1, 10001), t1, d, period=1)
    dmax = np.max(np.abs(d))

    # Plot results
    # matplotlib.use('PDF')
    lw = 1
    s = 1
    plot.plotopen('Frame ' + str(k), 1)
    plt.imshow(x, cmap='gray', vmin=0, vmax=2)
    # plt.plot(c1p[0], c1p[1], 'g', zorder=50, lw=lw)
    # plt.plot(c2p[0], c2p[1], 'b', zorder=100, lw=lw)
    plt.colorbar(plt.scatter(c1p[0], c1p[1], c=d, cmap='bwr', vmin=-dmax, vmax=dmax, zorder=50, s=lw), label='Displacement [pixels]')
    # plt.scatter(c2p[0], c2p[1], 'b', zorder=100, lw=lw)
    for j in range(len(t2)):
        plt.arrow(c1[0][j], c1[1][j], s*(c2[0][j] - c1[0][j]), s*(c2[1][j] - c1[1][j]), color='g', zorder=200, lw=lw)
        # plt.arrow(c1[0][j], c1[1][j], s * d1[0][j], s * d1[1][j], color='y', zorder=200, lw=lw)
    plt.arrow(c1[0][0], c1[1][0], s*(c2[0][0] - c1[0][0]), s*(c2[1][0] - c1[1][0]), color='c', zorder=400, lw=lw)
    for j in range(w.shape[0]):
        for i in range(int(ncurv/2**j)):
            if np.any(w[j, i]):
                p = center_of_mass(w[j, i])
                plt.text(p[1], p[0], str(i), color='yellow', fontsize=4, horizontalalignment='center', verticalalignment='center')
    plot.plotclose(False)
    # plot.show()


def discretizeCurve(tck, o):
    """ DEPRECATED - Convert a continuous spline to a discrete contour. """
    K = 10001
    cp = splev(np.mod(np.linspace(o, o + 1, K), 1), tck)
    cp = [np.round(cp[0]).astype(np.int), np.round(cp[1]).astype(np.int)]
    c = [np.asarray([cp[0][0], cp[1][0]])]
    n = 0
    for k in range(1, K):
        p = np.asarray([cp[0][k], cp[1][k]])
        if 0 < norm(c[n]-p):
            c.append(p)
            n += 1
    c.pop()
    c = np.asarray(c)
    return c

def rasterizeCurve(shape, s):
    # Construct a mapping from edge pixels to spline arguments
    delta = np.inf * np.ones(shape)
    tau = - np.ones(shape)
    t = np.linspace(0, 1, 10001)
    p = np.asarray(splev(np.mod(t, 1), s))
    pr = np.round(p)
    pi = pr.astype(dtype=np.int)
    for n in range(10001):
        d0 = np.linalg.norm(p[:,n]-pr[:,n])
        if d0 < delta[pi[1,n], pi[0,n]]:
            delta[pi[1, n], pi[0, n]] = d0
            tau[pi[1, n], pi[0, n]] = t[n]
    return tau

# # Load images
# path = 'C:\\Work\\UniBE 2\\Guillaume\\Example_Data\\FRET_sensors + actin\\Histamine\\Expt2\\w16TIRF-CFP\\'
# x1 = imread(path + 'RhoA_OP_his_02_w16TIRF-CFP_t71.tif')
# x2 = imread(path + 'RhoA_OP_his_02_w16TIRF-CFP_t131.tif')
#
# # Segment images
# c1, mask1 = segment(x1)
# c2, mask2 = segment(x2)
#
# mapContours(x1, c1, c2)