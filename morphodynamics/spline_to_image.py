import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import convolve2d

from .displacementestimation import compute_curvature

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
    pi = np.round(p).astype(dtype=np.int)
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
    pi = np.round(p).astype(dtype=np.int)
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
    if thickness > 1:
        f = np.ones((thickness, thickness))
        n = convolve2d(mask, f, mode="same")
        mask = n > 0
        n[np.logical_not(mask)] = 1
        im_contour = convolve2d(im_contour, f, mode="same") / n

    return im_contour

def edge_colored_by_displacement(data, res, t, N=None, cumulative=False, enlarge_width=1):

    if N is None:
        N = res.u[t]

    if cumulative:
        disp = np.cumsum(res.displacement, axis=1)
    else:
        disp = res.displacement
    im_contour_bw = spline_to_param_image(N=N, im_shape=data.shape, s=res.spline[t], deltat=0)
    im_coloured, mask = colour_image_border_by_feature(
        im_contour_bw=im_contour_bw, t_param=res.param0[t+1], feature=disp[:,t])
    im_coloured = enlarge_contour(im_coloured, mask, enlarge_width)
    return im_coloured

def edge_colored_by_curvature(data, res, t, N=None, enlarge_width=1):

    if N is None:
        N = res.u[t]

    t_param = np.linspace(0, 1, N, endpoint=False)
    curv = compute_curvature(res.spline[t], t_param)
    im_contour_bw = spline_to_param_image(N=N, im_shape=data.shape, s=res.spline[t], deltat=0)
    im_coloured, mask = colour_image_border_by_feature(
        im_contour_bw=im_contour_bw, t_param=t_param, feature=curv)
    im_coloured = enlarge_contour(im_coloured, mask, enlarge_width)
    return im_coloured