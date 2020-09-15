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
    """Create an image where the sampling windows are shown as regions with unique gray levels."""
    
    tiles = np.zeros(shape, dtype=np.uint16)
    n = 1
    for j in range(len(windows)):
        for i in range(len(windows[j])):
            tiles[windows[j][i][0], windows[j][i][1]] = n
            n += 1
    return tiles


def compute_discrete_arc_length(c):
    """Compute cumulative arc length along contour"""

    L = np.cumsum(
        np.linalg.norm(
            np.diff(np.concatenate([[c[0]], c, [c[0]]]), axis=0),
            axis=1)
        )
    #N = c.shape[0]
    #L = np.zeros((N+1,))
    #for n in range(1, N):
    #    L[n] = L[n-1] + np.linalg.norm(c[n]-c[n-1])
    #L[N] = L[N-1] + np.linalg.norm(c[0]-c[N-1])
    return L


def create_arc_length_image(shape, c, L):
    """Create image of cumulative arc length by assigning cumulative length to
    pixels along contour"""

    x = -np.ones(shape)
    x[c[np.arange(c.shape[0]), 0], c[np.arange(c.shape[0]), 1]] = L[np.arange(c.shape[0])]
    #for n in range(c.shape[0]):
    #    x[c[n,0], c[n,1]] = L[n]
    return x


# def define_contour_positions(L, I, cvec, cim):
#     t = np.zeros((I,))
#     for i in range(I):
#         L0 = (L[-1] / I) * (i + 0.5)
#         n = np.argmin(np.abs(L-L0))
#         t[i] = cim[cvec[n, 0], cvec[n, 1]]
#     return t


def create_windows(c_main, origin, J=None, I=None, depth=None, width=None):
    """Create windows based on contour and windowing parameters.

    Note: to define the windows, this function uses pseudo-radial and pseudo-angular
    coordinates. The pseudo-radial coordinate is based on the distance transform of
    the rasterized version of the continuous spline that defines the contour of the
    cell. The pseudo-angular coordinate for layer j is based on the distance transform
    of the discrete contour of layer j. So there is a bit of an inconsistency between
    continuous and discrete contours.

    Parameters:
        c_main: A rasterized version of the contour, as obtained by rasterize_curve.
        origin: (y, x) coordinates of the origin of the curve.
        J: Number of window layers.
        I: Vector of dimension J specifying the number of windows per layer.
        depth: Desired depth of the windows.
        width: Desired width of the windows.

    Returns:
        If I and J are specified, a list of lists w such that w[j][i] is an array of all pixel positions that belong to window (j, i). If I and J are not specified, they are determined based on the depth and width arguments and the tuple w, J, I is returned.
    """

    origin = [origin[1], origin[0]]

    # plt.figure()
    # plt.imshow(c_main, 'gray', vmin=-1, vmax=1)
    # plt.plot(origin[1], origin[0], 'or')

    # Compute the distance transform of the main contour
    D_main = distance_transform_edt(-1 == c_main)

    # Compute the mask corresponding to the main contour
    mask_main = binary_fill_holes(-1 < c_main)

    # Divide the radial coordinate into J layers with specified depth
    Dmax = np.amax(D_main * mask_main)
    if J is None:
        J = int(math.ceil(Dmax / depth))
    b = np.linspace(0, Dmax, J + 1)  # Boundaries of the layers in terms of distances to the main contour

    if I is None:
        compute_num_win = True
        I = []
    else:
        compute_num_win = False
    w = []
    for j in range(J):
        w.append([])

        # The mask containing the interior of the cell starting from the j-th layer
        mask = (b[j] <= D_main) * mask_main

        # Extract the contour of the mask
        cvec = np.asarray(find_contours(mask, 0, fully_connected='high')[0], dtype=np.int)

        # Lvec = compute_discrete_arc_length(cvec)
        # c = create_arc_length_image(mask.shape, cvec, Lvec)
        # plt.figure()
        # plt.imshow(c, 'gray', vmin=-Lvec[-1], vmax=Lvec[-1])
        # plt.plot(origin[1], origin[0], 'or')
        # # plt.show()

        # Adjust the origin of the contour:
        # on the discrete contour cvec, find the closest point to the origin,
        # then apply a circular shift to cvec to make this point the first one.
        n0 = np.argmin(np.linalg.norm(cvec - origin, axis=1))
        cvec = np.roll(cvec, -n0, axis=0)

        # Compute the discrete arc length along the contour
        Lvec = compute_discrete_arc_length(cvec)

        # Create an image of the contour where the intensity is the arc length
        c = create_arc_length_image(mask.shape, cvec, Lvec)
        # plt.figure()
        # plt.imshow(c, 'gray', vmin=-Lvec[-1], vmax=Lvec[-1])
        # plt.plot(origin[1], origin[0], 'or')
        # # plt.show()

        # Compute the feature transform of this image:
        # for each pixel position, we get the coordinates of the closest pixel on the contour
        F = distance_transform_edt(-1 == c, return_distances=False, return_indices=True)

        # Fill array with arc lengths of closest points on the contour
        #L = np.zeros(c.shape)
        #for u in range(c.shape[0]):
        #    for v in range(c.shape[1]):
        #        L[u, v] = c[F[0, u, v], F[1, u, v]]

        # gridx, gridy = np.meshgrid(range(c.shape[1]), range(c.shape[0]))
        # L = c[F[0,:,:][gridy, gridx], F[1,:,:][gridy, gridx]]

        L = c[F[0,:,:], F[1,:,:]]

        # Create sampling windows for the j-th layer
        if compute_num_win:
            I.append(int(math.ceil(Lvec[-1] / width)))
        s = np.linspace(0, Lvec[-1], I[j] + 1)
        for i in range(I[j]):
            # w[-1].append(np.where(mask & (s1[i] <= L) & (L < s1[i+1]) & (b[0] <= D) & (D < b[1])))
            w[-1].append(np.where(mask & (s[i] <= L) & (L < s[i+1]) & (b[j] <= D_main) & (D_main < b[j+1])))
            # plt.figure()
            # plt.imshow(w[j][i])
            # plt.show()

        # # Compute positions on the contour that will be used for the displacement estimation
        # if j == 0:
        #     t = define_contour_positions(Lvec, I[0], cvec, c_main)

    # plt.figure()
    # plt.imshow(find_boundaries(label_windows(c_main.shape, w)), 'gray')
    # plt.plot(origin[1], origin[0], 'or')
    # plt.show()

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
    """ Extract the mean and variance of an image over the sampling windows.

    Parameters:
        y: Image.
        w: Windows.

    Returns: Tuple with mean and variance.
    """

    # Number of windows
    J = len(w)
    I = [len(e) for e in w]
    Imax = np.max(I)

    # Initialization of the mean and variance structures
    # Remember that the number of windows depends on the layer j.
    # Here we use rectangular arrays to store the mean and variance,
    # but some elements will be unused and have a NaN value.
    mean = np.nan * np.ones((J,Imax))
    var = np.nan * np.ones((J,Imax))

    # Extraction of the mean and variance
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


def calculate_windows_index(w):
    """ Display the sampling-window boundaries and indices. """

    windows_pos = []
    for j in range(len(w)):
        for i in range(len(w[j])):
            if np.any(w[j][i]):
                p = [np.mean(w[j][i][0]), np.mean(w[j][i][1])]
                windows_pos.append([p[1], p[0], i])

    return windows_pos
