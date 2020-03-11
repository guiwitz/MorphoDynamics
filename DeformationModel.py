from scipy.signal import bspline
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.external.tifffile import imread
import math
from ArtifactGenerator import Plot
# from cython.hello import fc

# Small tests
def bsplineTest():
    x = np.linspace(-5, 5, 10001)
    plt.plot(x, bspline(x, 2))
    plt.show()

def tensorProductTest():
    a = np.arange(1,4).reshape((3,1))
    b = np.arange(4,7).reshape((1,3))
    c = a * b
    print(c)

# Data
# Biological cell
# b = imread('C:\\Work\\UniBE 2\\Guillaume\\Example_Data\\FRET_sensors + actin\\Histamine\\Expt2\\w16TIRF-CFP\\RhoA_OP_his_02_w16TIRF-CFP_t1.tif')
# b = b[0:-1:10, 1:-1:10] # Subsample image if desired

# Artificial grid
b = np.zeros((41,41))
b[0::10, :] = 1
b[:, 0::10] = 1

# Parameters
N = len(b.shape) # Number of dimensions
nm = 3 # Spline degree
h = 1 # Subsampling factor for the coordinate deformation

# Coordinates
# Corner origin
x0 = np.arange(0, b.shape[0], dtype=float)
x1 = np.arange(0, b.shape[1], dtype=float)
x0, x1 = np.meshgrid(x0, x1)
x = np.stack([x0, x1])
# Center origin
y0 = np.arange(-b.shape[0] / 2, b.shape[0] / 2, dtype=float)
y1 = np.arange(-b.shape[1] / 2, b.shape[1] / 2, dtype=float)
y0, y1 = np.meshgrid(y0, y1)
y = np.stack([y0, y1])

# Basis functions
def beta(x):
    b = 1
    for k in range(0, N):
        b = b * bspline(x[k], nm)
    return b

def beta2(x0, x1):
    b = bspline(x0, nm)
    b = b * bspline(x1, nm)
    return b

def phi(x):
    return beta(x/h)

# Coordinate deformation
def g(c, x):
    beta = phi(y)
    beta = np.fft.ifftshift(beta)
    betahat = np.fft.fftn(beta)
    return x - np.stack([np.real(np.fft.ifftn(np.fft.fftn(c[0]) * betahat)), np.real(np.fft.ifftn(np.fft.fftn(c[1]) * betahat))])
    # return x - np.stack([convolve2d(c[0], beta, mode='same'), convolve2d(c[1], beta, mode='same')])
    # return x - c

# Interpolated image
def fc(b, x):
    f = np.zeros(b.shape)
    for j0 in range(b.shape[0]):
        for j1 in range(b.shape[1]):
            # i0min = max(0, math.ceil(x[0,j0,j1]-(nm+1)/2))
            # i0max = min(b.shape[0]-1, math.floor(x[0,j0,j1]+(nm+1)/2))
            # # i0min = math.ceil(x[0,j0,j1]-(nm+1)/2)
            # # i0max = math.floor(x[0,j0,j1]+(nm+1)/2)
            # for i0 in range(i0min, i0max+1):
            #     i1min = max(0, math.ceil(x[1, j0, j1] - (nm + 1) / 2))
            #     i1max = min(b.shape[1]-1, math.floor(x[1, j0, j1] + (nm + 1) / 2))
            #     # i1min = math.ceil(x[1, j0, j1] - (nm + 1) / 2)
            #     # i1max = math.floor(x[1, j0, j1] + (nm + 1) / 2)
            #     for i1 in range(i1min, i1max+1):
            #         # if (abs(x0[j0,j1] - i0) < (nm + 1) / 2) & (abs(x1[j0,j1] - i1) < (nm + 1) / 2):
            #             f[j0,j1] += b[i0,i1] * beta(x[:,j0,j1] - [i0,i1])

            # i0min = max(0, math.ceil(x[0,j0,j1]-(nm+1)/2))
            # i0max = min(b.shape[0]-1, math.floor(x[0,j0,j1]+(nm+1)/2))
            # i1min = max(0, math.ceil(x[1, j0, j1] - (nm + 1) / 2))
            # i1max = min(b.shape[1] - 1, math.floor(x[1, j0, j1] + (nm + 1) / 2))
            # i0, i1 = np.meshgrid(range(i0min, i0max+1), range(i1min, i1max+1))
            # f[j0,j1] = np.sum(b[i0,i1] * beta(np.stack([x[0,j0,j1]-i0, x[1,j0,j1]-i1])))

            i0min = max(0, math.ceil(x[0,j0,j1]-(nm+1)/2))
            i0max = min(b.shape[0]-1, math.floor(x[0,j0,j1]+(nm+1)/2))
            i1min = max(0, math.ceil(x[1, j0, j1] - (nm + 1) / 2))
            i1max = min(b.shape[1] - 1, math.floor(x[1, j0, j1] + (nm + 1) / 2))
            i0, i1 = range(i0min, i0max+1), range(i1min, i1max+1)
            i0 = np.reshape(i0, (len(i0), 1))
            i1 = np.reshape(i1, (1, len(i1)))
            f[j0,j1] = np.sum(b[i0,i1] * beta2(x[0,j0,j1]-i0, x[1,j0,j1]-i1))

            # i0 = int(x[0, j0, j1])
            # i1 = int(x[1, j0, j1])
            # if (0 <= i0) & (i0 < b.shape[0]) & (0 <= i1) & (i1 < b.shape[1]):
            #     f[j0,j1] += b[i0,i1]
    return f

p = Plot(True)

# Image without deformation
c = np.zeros((2,) + b.shape)
f = fc(b, g(c, x))
# plt.figure()
# plt.imshow(f, cmap='gray')
p.imshow('Without deformation', f)

# Image with deformation
# c = np.stack([10*np.ones(b.shape), np.zeros(b.shape)])
# c = np.stack([np.sin(np.sqrt(x0tmp**2 + x1tmp**2)/3)*xtmp[0], np.sin(np.sqrt(x0tmp**2 + x1tmp**2)/3)*xtmp[1]])
c = y / np.amax(y) * 5
g0 = g(c, x)
f = fc(b, g0)
g0 = x - g0
p.plotopen('Deformation field')
plt.quiver(g0[0], -g0[1])
plt.axis('image')
plt.gca().invert_yaxis()
p.plotclose()
p.imshow('With deformation', f)
p.show()