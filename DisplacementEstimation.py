from skimage.external.tifffile import imread, imsave
# from skimage.exposure import histogram
# from skimage.filters import gaussian
# from skimage.segmentation import find_boundaries
# from skimage.measure import find_contours
# from scipy.interpolate import UnivariateSpline
import matplotlib
import matplotlib.pyplot as plt
# from scipy.signal import argrelmin
from scipy.interpolate import splprep, splev
from scipy.optimize import least_squares
import numpy as np
# import math
from numpy.linalg import norm
from Segmentation import segment
from FunctionalDefinition import Functional
from ArtifactGenerator import Plot

plot = Plot(not True)

# # Load images
# path = 'C:\\Work\\UniBE 2\\Guillaume\\Example_Data\\FRET_sensors + actin\\Histamine\\Expt2\\w16TIRF-CFP\\'
# x1 = imread(path + 'RhoA_OP_his_02_w16TIRF-CFP_t71.tif')
# x2 = imread(path + 'RhoA_OP_his_02_w16TIRF-CFP_t131.tif')
#
# # Segment images
# c1, mask1 = segment(x1)
# c2, mask2 = segment(x2)

def fitSpline(c):
    # Fit spline to contour
    # s = 0
    s = 1e2
    # s = 1e3
    # tck, u = splprep([[0, 50, 50, 0], [0, 0, 100, 100]], s=0, per=4)
    tck, u = splprep([c[:, 1], c[:, 0]], s=s, per=c.shape[0])
    return tck

def convolve(x, y):
    z = np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(y[::-1])))
    return z

def mapContours(tck1, tck2, p):
    # t = np.linspace(0, 1, 10000, endpoint=False)
    # s1 = np.asarray(splev(t, tck1))
    # s2 = np.asarray(splev(t, tck2))
    # s1m = np.mean(s1, axis=1)
    # s2m = np.mean(s2, axis=1)
    # t0 = t[np.argmax(convolve(s1[0]-s1m[0], s2[0]-s2m[0]) + convolve(s1[1]-s1m[1], s2[1]-s2m[1]))]
    # print(t0)
    #
    # p = p - t0
    # # o = np.mod(p - t0, 1)

    # Positions of the velocity arrows
    n = len(p)
    o = p

    # Weight for the cost function
    # w = 0
    w = np.sum((np.concatenate(splev(o, tck2)) - np.concatenate(splev(p, tck1)))**2) / (n-1)
    # w = np.sum(1 / (np.concatenate(splev(o, tck2)) - np.concatenate(splev(p, tck1)))**2) * (n-1)
    # w = 1e6

    # Lower and upper bounds for the least-squares problem
    lb = np.zeros((n+1,))
    lb[0] = -np.inf
    lb[n] = -1
    ub = np.inf * np.ones((n+1,))

    # Solve least-squares problem
    functional = Functional(tck1, tck2, p, w)
    result = least_squares(functional.f, functional.transform(o), bounds=(lb,ub), ftol=1e-3)
    o = functional.inversetransform(result.x)
    # o = p

    return o

# def alignOrigins(s0, s)

def plotMap(x, tck1, tck2, p, o,):
    # Evaluate splines at various points
    c1 = splev(np.mod(p, 1), tck1)
    c2 = splev(np.mod(o, 1), tck2)
    # c2a = splev(p, tck2)
    c1p = splev(np.linspace(0, 1, 10001), tck1)
    c2p = splev(np.linspace(0, 1, 10001), tck2)

    # Plot results
    # matplotlib.use('PDF')
    lw = 1
    s = 1

    plt.imshow(x, cmap='gray')
    plt.plot(c1p[0], c1p[1], 'g', zorder=50, lw=lw)
    plt.plot(c2p[0], c2p[1], 'b', zorder=100, lw=lw)
    for j in range(len(o)):
        # plt.arrow(c1[0][j], c1[1][j], c2a[0][j] - c1[0][j], c2a[1][j] - c1[1][j], color='c', zorder=200, lw=lw)
        plt.arrow(c1[0][j], c1[1][j], s*(c2[0][j] - c1[0][j]), s*(c2[1][j] - c1[1][j]), color='r', zorder=200, lw=lw)
        # plt.arrow(c1[0][j], c1[1][j], s * d1[0][j], s * d1[1][j], color='r', zorder=200, lw=lw)
    plt.arrow(c1[0][0], c1[1][0], s*(c2[0][0] - c1[0][0]), s*(c2[1][0] - c1[1][0]), color='c', zorder=400, lw=lw)

def discretizeCurve(tck, o):
    # Convert a continuous spline to a discrete contour
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

# mapContours(x1, c1, c2)