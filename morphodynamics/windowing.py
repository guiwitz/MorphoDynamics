import math

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.segmentation import find_boundaries


def label_windows(shape, windows):
    """
    Create an image where the sampling windows are shown as regions
    with unique gray levels.

    Parameters
    ----------
    shape: tuple
        intended shape of image
    windows: 3d list
        list of window indices as output by create_windows()

    Returns
    -------
    tiles: 2d array
        array where individual windows are labelled with unique index

    """

    tiles = np.zeros(shape, dtype=np.uint16)
    n = 1
    for j in range(len(windows)):
        for i in range(len(windows[j])):
            tiles[windows[j][i][0], windows[j][i][1]] = n
            n += 1
    return tiles


def compute_discrete_arc_length(c):
    """
    Compute cumulative arc length along contour c.

    Parameters
    ----------
    c: 2d array
        list of contour coordinates

    Returns
    -------
    L: 1d array
        arc length along contour

    """

    L = np.cumsum(
        np.linalg.norm(np.diff(np.concatenate([[c[0]], c, [c[0]]]), axis=0), axis=1)
    )
    return L


def create_arc_length_image(shape, c, L):
    """
    Create image of cumulative arc length by assigning cumulative length to
    pixels along contour.

    Parameters
    ----------
    shape: tuple
        intended shape of image
    c: 2d array
        contour coordinates
    L: 2d array
        arc length along contour

    """

    x = -np.ones(shape)
    x[c[np.arange(c.shape[0]), 0], c[np.arange(c.shape[0]), 1]] = L[
        np.arange(c.shape[0])
    ]

    return x


def create_windows(c_main, origin, J=None, I=None, depth=None, width=None):
    """
    Create windows based on contour and windowing parameters. The first
    window (at arc length = 0) is placed at the spline origin.

    Note: to define the windows, this function uses pseudo-radial and
    pseudo-angular coordinates. The pseudo-radial coordinate is based
    on the distance transform of the rasterized version of the continuous
    spline that defines the contour of the cell. The pseudo-angular coordinate
    for layer j is based on the distance transform of the discrete contour of
    layer j. So there is a bit of an inconsistency between continuous and
    discrete contours.

    Parameters
    ----------
    c_main: 2d array
        A rasterized version of the contour, as obtained 
        by spline_to_param_image.
    origin: tuple
        (y, x) coordinates of the origin of the curve.
    J: int
        Number of window layers.
    I: list of int
        Vector of dimension J specifying the number of windows per layer.
    depth: int
        Desired depth of the windows.
    width: int
        Desired width of the windows.

    Returns
    -------
    w: 3d list
        w[i][j][0] and w[i][j][1] are 1d arrays representing
        lists of x,y indices of pixels belonging to window in i'th layer
        in j'th window
    J: int
        number of layers (calculated if not provided as input)
    I: list of int
        number of windows per layer (calculated if not provided as input)

    """

    origin = [origin[1], origin[0]]

    # Compute the distance transform of the main contour
    D_main = distance_transform_edt(-1 == c_main)

    # Compute the mask corresponding to the main contour
    mask_main = binary_fill_holes(
        -1 < c_main
    )  # Maybe not necessary? Can't we just use the segmented mask here?

    # Divide the radial coordinate into J layers with specified depth
    Dmax = np.amax(D_main * mask_main)
    if J is None:
        J = int(math.ceil(Dmax / depth))
    b = np.linspace(
        0, Dmax, J + 1
    )  # Boundaries of the layers in terms of distances to the main contour

    if I is None:
        compute_num_win = True
        I = []
    else:
        compute_num_win = False
    w = []
    for j in range(J):
        w.append([])

        # The mask containing the interior of the cell starting from
        # the j-th layer
        mask = (b[j] <= D_main) * mask_main

        # Extract the contour of the mask
        # We must fix certain frames where multiple contours are returned.
        # So we choose the longest contour. Some pixels may be lost in the process,
        # i.e., the windows may not cover the entire cell.
        clist = find_contours(mask, 0, fully_connected="high")
        cvec = np.asarray(
            clist[np.argmax([cel.shape[0] for cel in clist])], dtype=int
        )

        # An alternative fix using OpenCV's findContours routine---doesn't solve the problem
        # contours, hierarchy = cv.findContours(np.asarray(mask, dtype=np.uint8), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        # cvec = np.asarray(contours[np.argmax([cel.shape[0] for cel in contours])], dtype=int)
        # cvec = cvec.reshape((cvec.shape[0], cvec.shape[2]))
        # cvec = cvec[::-1, [1,0]]  # Sort boundary pixels in clockwise direction and switch (x, y) coordinates

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
        arc = create_arc_length_image(mask.shape, cvec, Lvec)

        # Compute the feature transform of this image:
        # for each pixel position, we get the coordinates of the closest pixel on the contour
        F = distance_transform_edt(
            -1 == arc, return_distances=False, return_indices=True
        )

        # Fill array with arc lengths of closest points on the contour
        # L = np.zeros(c.shape)
        # for u in range(c.shape[0]):
        #    for v in range(c.shape[1]):
        #        L[u, v] = c[F[0, u, v], F[1, u, v]]

        # gridx, gridy = np.meshgrid(range(c.shape[1]), range(c.shape[0]))
        # L = c[F[0,:,:][gridy, gridx], F[1,:,:][gridy, gridx]]

        L = arc[F[0, :, :], F[1, :, :]]

        # Create sampling windows for the j-th layer
        if compute_num_win:
            I.append(int(math.ceil(Lvec[-1] / width)))
        w_borders = np.linspace(0, Lvec[-1], I[j] + 1)
        for i in range(I[j]):
            # w[-1].append(np.where(mask & (s1[i] <= L) & (L < s1[i+1]) & (b[0] <= D) & (D < b[1])))
            w[-1].append(
                np.where(
                    mask
                    & (w_borders[i] <= L)
                    & (L < w_borders[i + 1])
                    & (b[j] <= D_main)
                    & (D_main < b[j + 1])
                )
            )
            # plt.figure()
            # plt.imshow(w[j][i])
            # plt.show()

        # # Compute positions on the contour that will be used for the displacement estimation
        # if j == 0:
        #     t = define_contour_positions(Lvec, I[0], cvec, c_main)

    return w, J, I


def extract_signals(y, w):
    """
    Extract the mean and variance of an image over the sampling windows.

    Parameters
    ----------
    y: 2d array, Image
    w: 3d list
        list of window indices as output by create_windows()

    Returns
    -------
    mean : 2d array
        mean values of signal in each window. Array of size number of
        layers times number of windows in outer layer
    var : 2d array
        variance values of signal in each window. Array of size number of
        layers times number of windows in outer layer

    """

    # Number of windows
    J = len(w)
    I = [len(e) for e in w]
    Imax = np.max(I)

    # Initialization of the mean and variance structures
    # Remember that the number of windows depends on the layer j.
    # Here we use rectangular arrays to store the mean and variance,
    # but some elements will be unused and have a NaN value.
    mean = np.nan * np.ones((J, Imax))
    var = np.nan * np.ones((J, Imax))

    # Extraction of the mean and variance
    for j in range(J):
        for i in range(I[j]):
            mean[j, i] = np.mean(y[w[j][i][0], w[j][i][1]])
            var[j, i] = np.var(y[w[j][i][0], w[j][i][1]])
    return mean, var


def show_windows(w, b):
    """Display the sampling-window boundaries and indices."""

    plt.imshow(b, cmap="gray", vmin=0, vmax=2)
    for j in range(len(w)):
        for i in range(len(w[j])):
            if np.any(w[j][i]):
                p = [np.mean(w[j][i][0]), np.mean(w[j][i][1])]
                plt.text(
                    p[1],
                    p[0],
                    str(i),
                    color="yellow",
                    fontsize=4,
                    horizontalalignment="center",
                    verticalalignment="center",
                )


def boundaries_image(im_shape, window):
    """
    Create array where windows boundaries are set to 1 and
    the background to nan

    Parameters
    ----------
    im_shape: tuple
        size of image to produce
    window: 3d list
        window indices list as produced by create_windows()
    """
    b0 = find_boundaries(label_windows(im_shape, window))
    b0 = b0.astype(float)
    b0[b0 == 0] = np.nan

    return b0


def calculate_windows_index(w):
    """
    Calculate per window index position.

    Parameters
    ----------
    w: 3d list
        list of window indices as output by create_windows()

    Returns
    -------
    windows_pos: list of lists
        each element is a list [x, y, index] where x,y are the
        window average position

    """

    windows_pos = []
    for j in range(len(w)):
        for i in range(len(w[j])):
            if np.any(w[j][i]):
                p = [np.mean(w[j][i][0]), np.mean(w[j][i][1])]
                windows_pos.append([p[1], p[0], i])

    return windows_pos
