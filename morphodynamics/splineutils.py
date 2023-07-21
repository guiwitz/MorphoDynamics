import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import convolve2d
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.measure import find_contours

from .windowing import compute_discrete_arc_length

def splevper(t, s_tuple):
    """
    Evaluate B-spline for periodic curve

    Parameters
    ----------
    t: 1d array
        points on which to return the evaluation
    s_tuple: spline tuple as returned by splprep

    Returns
    -------
    list of arrays
        coordinates of evaluated spline

    """
    return splev(np.mod(t, 1), s_tuple)

def fit_spline(c, lambda_):
    """ "
    Fit a spline to a contour specified as a list of pixels.

    Parameters
    ----------
    c: 2d array
        contour coordinates
    lambda_: float
        smoothing parameter (as used by splprep)

    Returns
    -------
    s: tuple
        spline tuple
    u: 1d array
        an array of the values of the parameter

    """

    s_tuple, u = splprep([c[:, 1], c[:, 0]], s=lambda_, per=c.shape[0])  # Fitting with periodic boundary conditions
    
    return s_tuple, u

def spline_contour_length(s_tuple, t1=0, t2=1, N=None):
    """
    Get spline contour length between params t1 and t2

    Parameters
    ----------
    t1, t2: float
        parameter limits
    N: int
        number of points on the contour
    s: tuple
        spline tuple as returned by splprep

    Returns
    -------
    spline_length: float
        contour length of spline

    """

    # to get a good approx. take 3 x the number of knots
    if N is None:
        N = 3 * len(s_tuple[0])

    spline = np.array(splev(np.linspace(t1, t2, N), s_tuple))
    lengths = np.sqrt(np.sum(np.diff(spline.T, axis=0)**2, axis=1))
    spline_length = np.sum(lengths)
    return spline_length

def spline_area(s_tuple, N=None):
    """Compute area of spline s discretized in N segments."""

    # to get a good approx. take 3 x the number of knots
    if N is None:
        N = 3 * len(s_tuple[0])

    c = splev(np.linspace(0, 1, N, endpoint=False), s_tuple)
    cprm = splev(np.linspace(0, 1, N, endpoint=False), s_tuple, der=1)
    
    return np.sum(c[0] * cprm[1] - c[1] * cprm[0]) / 2 / N


def spline_curvature(s_tuple, t):
    """Compute local curvature of spline s at paramters positions t."""
    
    cprm = splev(t, s_tuple, der=1)
    csec = splev(t, s_tuple, der=2)
    return (cprm[0] * csec[1] - cprm[1] * csec[0]) / (
        cprm[0] ** 2 + cprm[1] ** 2
    ) ** 1.5

def subdivide_curve(N, s, orig, I):
    """
    Define points on a contour that are equispaced with respect to the arc length.
    
    Parameters
    ----------
    N: int
        number of points for spline discretization
    s: tuple
        spline tuple as returned by splprep
    origin: float
        shift of parameter origin
    I: int
        Number of windows in the first (outer) layer.

    Returns
    -------
    t_shifted: 1d array
        list of spline parameters on s defining the same points as cvec_sel
    
    """

    t = np.linspace(0, 1, N + 1)
    #L = np.cumsum(np.linalg.norm(splevper(t + orig, s), axis=0))
    L = np.cumsum(np.linalg.norm(np.diff(np.stack(splevper(t, s)).T, axis=0), axis=1))
    t0 = np.zeros((I,))
    n = 0
    for i in range(I):
        p = L[-1] / I * (0.5 + i)
        while L[n] < p:
            n += 1
        t0[i] = t[n]
    t_shifted = t0 + orig
    return t_shifted

def subdivide_curve_discrete(N, c_main, I, s, origin):
    """
    Creates a discrete contour whose first pixel corresponds
    to the specified origin, plus a list of coordinates along the
    continuous curve corresponding to the mid-points of the
    windows in the first (outer) layer.

    Note: this function tries to reconcile discrete and continuous
    representations of the contour, so it may not be conceptually
    very satisfactory.

    Parameters
    ----------
    N: int
        number of points for spline discretization
    c_main: 2d array
        A rasterized version of the contour, as obtained by rasterize_curve.
    I: int
        Number of windows in the first (outer) layer.
    s: tuple
        spline tuple as returned by splprep
    origin: ndarray
        [y, x] coordinates of the origin of the curve.

    Returns
    -------
    cvec_sel: 2d array
        xy array of selected positions along the contour
    t_sel: 1d array
        list of spline parameters on s defining the same points as cvec_sel

    """

    origin = [origin[1], origin[0]]

    # Compute the distance transform of the main contour
    D_main = distance_transform_edt(-1 == c_main)

    # Compute the mask corresponding to the main contour
    mask_main = binary_fill_holes(-1 < c_main)

    # To be verified: this might actually be the same as mask_main
    mask = (0 <= D_main) * mask_main

    # Extract the contour of the mask
    cvec = np.asarray(find_contours(mask, 0, fully_connected="high")[0], dtype=int)

    # Adjust the origin of the contour:
    # on the discrete contour cvec, find the closest point to the origin,
    # then apply a circular shift to cvec to make this point the first one.
    n0 = np.argmin(np.linalg.norm(cvec - origin, axis=1))
    cvec = np.roll(cvec, -n0, axis=0)

    # Compute the discrete arc length along the contour
    Lvec = compute_discrete_arc_length(cvec)

    # Compute the index of the mid-point for each window
    # Note that the arc length is being used as a coordinate along the curve
    n = np.zeros((I,), dtype=int)
    for i in range(I):
        n[i] = np.argmin(np.abs(Lvec - Lvec[-1] / I * (0.5 + i)))
    cvec_sel = cvec[n, :]

    # Compute the parameter of the first mid-point
    t = np.linspace(0, 1, N, endpoint=False)
    c = splevper(t, s)
    m = np.argmin(np.linalg.norm(np.transpose(c) - np.flip(cvec[n[0]]), axis=1))

    # Convert the index along the discrete contour to a position along the continuous contour
    # When searching for the closest spline position to a window, remove already "used" locations
    # so that the path does not come back on itself
    t = np.linspace(t[m], t[m] + 1, N, endpoint=False)
    c = splevper(t, s)
    m = np.zeros((I,), dtype=int)
    for i in range(I):
        m[i] = np.argmin(np.linalg.norm(np.transpose(c) - np.flip(cvec[n[i]]), axis=1))
        c = [c[0][m[i]+1::], c[1][m[i]+1::]]
    m = m+1
    m[0] = 0
    m = np.cumsum(m)
    t_sel = t[m]

    return cvec_sel, t_sel

def spline_int_coordinates(N, s):
    """
    Get integer xy coordinates of a spline.

    Parameters
    ----------
    N: int
        number of points on the contour
    s: tuple
        spline tuple as returned by splprep

    Returns
    -------
    spline_int_xy: 2d array
        coordinates of spline

    """

    # create contour parameter and estimate position along contour and round it
    t = np.linspace(0, 1, N + 1)
    p = np.asarray(splev(t, s))  # The points on the curve
    pi = np.round(p).astype(dtype=int)
    return pi

def spline_to_binary_image(N, im_shape, s):
    """
    Turn a spline into a binary image.

    Parameters
    ----------
    N: int
        number of points on the contour
    im_shape: tuple
        size of the desired image.
    s: tuple
        spline tuple as returned by splprep

    Returns
    -------
    im_spline: 2d array
        rasterized image of the contour

    """

    # get spline xy coordinates
    pi = spline_int_coordinates(N, s)
    
    # create binary image
    im_spline = np.zeros(im_shape)
    im_spline[pi[0], pi[1]] = 1

    return im_spline

def spline_to_param_image(N, im_shape, s, deltat):
    """
    Represent a contour as a grayscale image.
    If a pixel is part of the contour, then its intensity
    is equal to the parameter t of the closest point on the contour s(t).
    Otherwise it is equal to -1.

    Parameters
    ----------
    N: int
        number of points on the contour
    im_shape: tuple
        size of the desired image.
    s: tuple
        spline tuple as returned by splprep
    deltat: float
        origin shift of the spline.

    Returns
    -------
    tau: 2d array
        rasterized image of the contour

    """

    # store distance to the closest point on the contour
    delta = np.inf * np.ones(im_shape)
    # store closest paramter t per pixel. -1 means not par of the contour
    tau = -np.ones(im_shape)
    # create contour parameter and estimate position along contour and round it
    t = np.linspace(0, 1, N + 1)
    p = np.asarray(splev(t, s))  # The points on the curve
    pi = np.round(p).astype(dtype=int)
    # shift to origin if necessary
    t = np.mod(t - deltat, 1)
    # computer distance between contour point and closest pixel
    d0 = np.linalg.norm(p - pi, axis=0)
    # multiple continuous points p can have the same integer pixels
    # for each pixel find the closest associated point and store its correponding t
    for n in range(N + 1):
        if (d0[n] < delta[pi[1, n], pi[0, n]]):
            delta[pi[1, n], pi[0, n]] = d0[n]
            tau[pi[1, n], pi[0, n]] = t[n]
    return tau

def colour_image_border_by_feature(im_contour_bw, t_param, feature):
    """
    Given an image of a contour colored by distance along contour,
    color the border of the image by the feature.

    Parameters
    ----------
    im_contour_bw : ndarray
        image of contour
    t_param : array
        parametric points along contour [0,1] of length N
    feature : array
        variable to be used for colouring the border of length N
        feature[i] is the value of the feature at t_param[i]

    Returns
    -------
    im_contour_bw
        image of contour colored by feature
    """

    mask = im_contour_bw > -1
    interpolated_values = np.interp(im_contour_bw[mask], t_param, feature, period=1)
    im_contour_bw[mask] = interpolated_values
    im_contour_bw[np.logical_not(mask)] = 0

    return im_contour_bw, mask

def enlarge_contour(im_contour, mask, thickness):
    """
    Given a single pixel gray-scale contour, expand it to give it
    a thickness.

    Parameters
    ----------
    im_contour : ndarray
        image of contour
    mask : ndarray
        maks of contour
    thickness : int
        desired contour thickness

    Returns
    -------
    im_contour, mask
        image of enlarged contour and its mask
    """
    
    if thickness > 1:
        f = np.ones((thickness, thickness))
        n = convolve2d(mask, f, mode="same")
        mask = n > 0
        n[np.logical_not(mask)] = 1
        im_contour = convolve2d(im_contour, f, mode="same") / n

    return im_contour, mask


def edge_colored_by_features(data, res, t, feature, N=None, enlarge_width=1):
    """Create gray-scale image of contour where pixel values are interpolated
    according to the feature.

    Parameters
    ----------
    data : data object
    res : result object
    t : int
        frame
    feature : str
        feature for coloring 'displacement', 'displacement_cumul', 'curvature'
    N : int
        number of points for contour generation, default None
    enlarge_width : int, optional
        width of contour for display, by default 1

    Returns
    -------
    im_coloured, mask
        colored image (gray-scale) and mask of (wide) contour pixels

    """

    if N is None:
        N = 3 * int(spline_contour_length(res.spline[t]))
        #N = 3*len(res.spline[t][0])

    if feature == 'curvature':
        t_param = np.linspace(0, 1, N, endpoint=False)
        f = spline_curvature(res.spline[t], t_param)
    elif feature == 'displacement':
        t_param = res.param0[t+1]
        f = res.displacement[:,t]
    elif feature == 'displacement_cumul':
        t_param = res.param0[t+1]
        f = np.cumsum(res.displacement, axis=1)[:,t]

    im_contour_bw = spline_to_param_image(N=N, im_shape=data.shape, s=res.spline[t], deltat=0)
    im_coloured, mask = colour_image_border_by_feature(
        im_contour_bw=im_contour_bw, t_param=t_param, feature=f)
    im_coloured, mask = enlarge_contour(im_coloured, mask, enlarge_width)
    
    return im_coloured, mask
    
