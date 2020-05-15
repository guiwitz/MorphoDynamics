import math

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.segmentation import find_boundaries
from skimage.color import label2rgb
# from skimage.external.tifffile import imread, imsave
# from Segmentation import segment
# from ArtifactGeneration import FigureHelper

# plot = FigureHelper(not True)


def label_windows(shape, windows):
    """ Create an image where the sampling windows are shown as regions with unique gray levels. """
    tiles = np.zeros(shape, dtype=np.uint16)
    n = 1
    for j in range(len(windows)):
        for i in range(len(windows[j])):
            tiles[windows[j][i][0], windows[j][i][1]] = n
            n += 1
    return tiles

def compute_discrete_arc_length(c):
    N = c.shape[0]
    L = np.zeros((N+1,))
    for n in range(1, N):
        L[n] = L[n-1] + np.linalg.norm(c[n]-c[n-1])
    L[N] = L[N-1] + np.linalg.norm(c[0]-c[N-1])
    return L


def create_arc_length_image(shape, c, L):
    x = -np.ones(shape)
    for n in range(c.shape[0]):
        x[c[n,0], c[n,1]] = L[n]
    return x


def define_contour_positions(L, I, cvec, cim):
    t = np.zeros((I,))
    for i in range(I):
        L0 = (L[-1] / I) * (i + 0.5)
        n = np.argmin(np.abs(L-L0))
        t[i] = cim[cvec[n, 0], cvec[n, 1]]
    return t


def create_windows(c_main, origin, J=None, I=None, depth=None, width=None):
    origin = [origin[1], origin[0]]

    # Compute the distance transform of the main contour
    D_main = distance_transform_edt(-1 == c_main)

    # Compute the mask corresponding to the main contour
    mask_main = binary_fill_holes(-1 < c_main)

    # Divide the radial coordinate into J layers
    Dmax = np.amax(D_main * mask_main)
    if J is None:
        J = int(math.ceil(Dmax / depth))
    b = np.linspace(0, Dmax, J + 1)

    if I is None:
        compute_num_win = True
        I = []
    else:
        compute_num_win = False
    w = []
    for j in range(J):
        w.append([])

        mask = (b[j] <= D_main) * mask_main

        # Extract the contour of the mask
        cvec = np.asarray(find_contours(mask, 0, fully_connected='high')[0], dtype=np.int)

        # Adjust the origin of the contour
        n0 = np.argmin(np.linalg.norm(cvec - origin, axis=1))
        c = np.roll(cvec, -n0, axis=0)

        # Compute the discrete arc length along the contour and create an image of the contour where the intensity is the arc length
        # ctmp, Lmax = compute_discrete_arc_length_old(mask.shape, c)
        # plt.figure()
        # plt.imshow(ctmp, 'gray')

        Lvec = compute_discrete_arc_length(c)
        c = create_arc_length_image(mask.shape, c, Lvec)
        # plt.figure()
        # plt.imshow(c, 'gray')
        # plt.show()

        # Compute the distance and feature transforms of this image
        F = distance_transform_edt(-1 == c, return_distances=False, return_indices=True)

        # Fill array with arc lengths of closest points on the contour
        L = np.zeros(c.shape)
        for u in range(c.shape[0]):
            for v in range(c.shape[1]):
                L[u, v] = c[F[0, u, v], F[1, u, v]]

        # Create sampling windows for the j-th layer
        if compute_num_win:
            I.append(int(math.ceil(Lvec[-1] / width)))
        s = np.linspace(0, Lvec[-1], I[j] + 1)
        for i in range(I[j]):
            # w[-1].append(np.where(mask & (s[i] <= L) & (L < s[i+1]) & (b[0] <= D) & (D < b[1])))
            w[-1].append(np.where(mask & (s[i] <= L) & (L < s[i+1]) & (b[j] <= D_main) & (D_main < b[j+1])))
            # plt.figure()
            # plt.imshow(w[j][i])
            # plt.show()

        # Compute positions on the contour that will be used for the displacement estimation
        if j == 0:
            t = define_contour_positions(Lvec, I[0], cvec, c_main)

    # # Artifact generation
    # plt.figure()
    # # plt.title('Contour')
    # plt.imshow(c, 'gray', vmin=-1, vmax=1)
    # plt.axis('off')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig('Contour.pdf')
    #
    # # plt.figure()
    # # plt.title('Mask')
    # # plt.imshow(m.astype(np.int), 'gray')
    #
    # plt.figure()
    # # plt.title('Distance transform')
    # plt.imshow(D, 'gray')
    # plt.axis('off')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig('Distance transform.pdf')
    #
    # # plot.imshow('Contour length', l)
    #
    # plt.figure()
    # # plt.title('Sectors')
    # plt.imshow(L, 'gray', vmin=0, vmax=1)
    # plt.axis('off')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig('Sectors.pdf')
    #
    # plt.figure()
    # # plt.title('Windows')
    # plt.imshow(label2rgb(label_windows(w)))
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('Windows.pdf')
    #
    # plt.figure()
    # # plt.title('Window boundaries')
    # plt.imshow(find_boundaries(label_windows(w)), 'gray')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('Window boundaries.pdf')

    # plt.figure()
    # plt.imshow(label2rgb(label_windows(mask.shape, w)))
    # plt.figure()
    # plt.imshow(find_boundaries(label_windows(mask.shape, w)), 'gray')
    # plt.show()

    if compute_num_win:
        return w, J, I
    else:
        return w


def extract_signals_old(y, w):
    """ Extract the mean values of an image over the sampling windows. """
    mean = np.nan * np.ones(w.shape[0:2])
    var = np.nan * np.ones(w.shape[0:2])
    for j in range(w.shape[0]):
        for i in range(w.shape[1]):
            if np.any(w[j, i]):
                mean[j, i] = np.mean(y[w[j, i]])
                var[j, i] = np.var(y[w[j, i]])
    return mean, var


def extract_signals(y, w):
    """ Extract the mean values of an image over the sampling windows. """
    J = len(w)
    I = [len(e) for e in w]
    Imax = np.max(I)
    mean = np.nan * np.ones((J,Imax))
    var = np.nan * np.ones((J,Imax))
    for j in range(J):
        for i in range(I[j]):
            mean[j, i] = np.mean(y[w[j][i][0], w[j][i][1]])
            var[j, i] = np.var(y[w[j][i][0], w[j][i][1]])
    return mean, var


def show_windows(w, b):
    """ Display the sampling-window boundaries and indices. """
    plt.imshow(b, cmap='gray', vmin=0, vmax=2)
    for j in range(len(w)):
        for i in range(len(w[j])):
            if np.any(w[j][i]):
                p = [np.mean(w[j][i][0]), np.mean(w[j][i][1])]
                plt.text(p[1], p[0], str(i), color='yellow', fontsize=4, horizontalalignment='center', verticalalignment='center')
