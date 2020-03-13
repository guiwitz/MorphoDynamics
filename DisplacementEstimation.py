import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.optimize import least_squares
import numpy as np
from numpy.linalg import norm

from FunctionalDefinition import Functional
from ArtifactGeneration import FigureHelper

plot = FigureHelper(not True)

def fitSpline(c):
    """" Fit a spline to a contour specified as a list of pixels. """
    # Smoothing parameter
    # lambd = 0
    lambd = 1e2
    # lambd = 1e3

    # Fitting with periodic boundary conditions
    s, u = splprep([c[:, 1], c[:, 0]], s=lambd, per=c.shape[0])

    return s

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

def showEdge(s1, s2, t1, t2, d, u):
    """ Draw the cell-edge contour and the displacement vectors. """

    # Evaluate splines at various points
    c1 = splev(np.mod(t1, 1), s1)
    c2 = splev(np.mod(t2, 1), s2)
    c1p = splev(np.linspace(0, 1, 10001), s1)
    c2p = splev(np.linspace(0, 1, 10001), s2)

    # Interpolate displacements
    # d = 0.5 + 0.5 * d / np.max(np.abs(d))
    d = np.interp(np.linspace(0, 1, 10001), t1, d, period=1)
    dmax = np.max(np.abs(d))

    # Plot results
    # matplotlib.use('PDF')
    lw = 1
    s = 1 # Scaling factor for the vectors

    # plt.plot(c1p[0], c1p[1], 'g', zorder=50, lw=lw)
    # plt.plot(c2p[0], c2p[1], 'b', zorder=100, lw=lw)
    plt.colorbar(plt.scatter(c1p[0], c1p[1], c=d, cmap='bwr', vmin=-dmax, vmax=dmax, zorder=50, s=lw), label='Displacement [pixels]')
    for j in range(len(t2)):
        plt.arrow(c1[0][j], c1[1][j], s*(c2[0][j] - c1[0][j]), s*(c2[1][j] - c1[1][j]), color='g', zorder=200, lw=lw)
        # plt.arrow(c1[0][j], c1[1][j], s * u[0][j], s * u[1][j], color='y', zorder=200, lw=lw)
    plt.arrow(c1[0][0], c1[1][0], s*(c2[0][0] - c1[0][0]), s*(c2[1][0] - c1[1][0]), color='c', zorder=400, lw=lw)

def rasterizeCurve(shape, s):
    """ Construct a mapping from edge pixels to spline arguments. """
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