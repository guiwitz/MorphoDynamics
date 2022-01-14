from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.optimize import least_squares, minimize, LinearConstraint
from scipy.signal import convolve2d
import numpy as np
from skimage.measure import find_contours

from matplotlib.backends.backend_pdf import PdfPages
from copy import deepcopy

from .functionaldefinition import Functional, Functional2, Functional3

# fh = FigureHelper(not True)
from .windowing import compute_discrete_arc_length


def splevper(t, s):
    """
    Evaluate B-spline for periodic curve

    Parameters
    ----------
    t: 1d array
        points on which to return the evaluation
    s: spline tuple as returned by splprep

    Returns
    -------
    list of arrays
        coordinates of evaluated spline

    """
    return splev(np.mod(t, 1), s)


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

    """

    s_tuple, u = splprep([c[:, 1], c[:, 0]], s=lambda_, per=c.shape[0])  # Fitting with periodic boundary conditions
    
    return s_tuple, u


def compute_length(N, s):
    """Compute length of spline s discretized in N segments."""

    cprm = splev(np.linspace(0, 1, N, endpoint=False), s, der=1)
    return np.sum(np.sqrt(cprm[0] ** 2 + cprm[1] ** 2)) / N


def compute_area(N, s):
    """Compute area of spline s discretized in N segments."""

    c = splev(np.linspace(0, 1, N, endpoint=False), s)
    cprm = splev(np.linspace(0, 1, N, endpoint=False), s, der=1)
    return np.sum(c[0] * cprm[1] - c[1] * cprm[0]) / 2 / N


def compute_curvature(s, t):
    """Compute local curvature of spline s at paramters positions t."""
    cprm = splev(t, s, der=1)
    csec = splev(t, s, der=2)
    return (cprm[0] * csec[1] - cprm[1] * csec[0]) / (
        cprm[0] ** 2 + cprm[1] ** 2
    ) ** 1.5


def correlate(x, y):
    """Compute the correlation between two signals with periodic boundary conditions."""
    z = np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(y[::-1])))
    return z


def find_origin(N, s1, s2, t0):
    """
    Find the parameter for s2 that best approximates the origin on s1
    given a shift t0.

    Parameters
    ----------
    N: int
        Number of samples along the curve
    s1: tuple
        spline tuple as returned by splprep
    s2: tuple
        spline tuple as returned by splprep
    t0: float
        Origin shift of curve s1

    Returns
    -------
    orig_param: float
        parameter for s2 matching origin position on s1

    """

    t = np.linspace(0, 1, N, endpoint=False)
    x = splev(t0, s1)
    c = splev(t, s2)
    n = np.argmin((c[0] - x[0]) ** 2 + (c[1] - x[1]) ** 2)
    orig_param = t[n]
    return orig_param


def align_curves(N, s1, s2, t1):
    """
    This function is intended to help improve the accuracy of displacement estimation
    when in addition to protrusions the cell is subject to motion.
    The idea is to find a translation and a change of origin for s2 that aligns it
    on s1.
    The translated curve s1c essentially accounts for the the motion of the cell, and
    once it is available, one can compute the usual displacement between s1c and s2.
    The total displacement is the sum of both components.

    Parameters
    ----------
    N: int
        Number of samples along the curve
    s1: tuple
        spline tuple as returned by splprep
    s2: tuple
        spline tuple as returned by splprep
    t1: float
        Origin of curve s1

    Returns
    -------
    s1c: bspline tuple
        translated curve s1 to match s2
    t2: float
        origin shift for s2 to match s1

    """

    t = np.linspace(0, 1, N, endpoint=False)

    def functional(v):
        """Computes the difference between s1c and s2 with adjusted origin.
        Used to minimize this difference using scipy.optimize.least_squares.

        Parameters:
            v: Three-dimensional vector representing the translation of s1 and the new origin of s2.
        """

        s1c = deepcopy(s1)
        s1c[1][0] += v[0]
        s1c[1][1] += v[1]
        t2 = v[2]
        c1 = splevper(t + t1, s1c)
        c2 = splevper(t + t2, s2)

        return np.concatenate(c2) - np.concatenate(c1)

    # Search for the optimal translation and change of origin
    lsq = least_squares(
        functional, [0, 0, t1], method="lm", x_scale=[1, 1, 1e-4]
    )  # , ftol=1e-3
    v = lsq.x

    # Construct the translated curve and the new origin
    s1c = deepcopy(s1)
    s1c[1][0] += v[0]
    s1c[1][1] += v[1]
    t2 = v[2]

    return s1c, t2


def map_contours(s1, s2, t1):
    """Compute displacement vectors between two consecutive contours."""

    # Positions of the velocity arrows
    N = len(t1)
    t2 = t1

    # Weight for the cost function
    # w = 0
    w = np.sum((np.concatenate(splev(t2, s2)) - np.concatenate(splev(t1, s1))) ** 2) / (
        N - 1
    )
    # w = np.sum(1 / (np.concatenate(splev(t2, tck2)) - np.concatenate(splev(t1, s1)))**2) * (N-1)
    # w = 1e6

    # Lower and upper bounds for the least-squares problem
    lb = np.zeros((N + 1,))
    lb[0] = -np.inf
    lb[N] = -1
    ub = np.inf * np.ones((N + 1,))

    # Solve least-squares problem
    functional = Functional(s1, s2, t1, w)
    result = least_squares(
        functional.f, functional.transform(t2), bounds=(lb, ub), ftol=1e-3
    )
    t2 = functional.inversetransform(result.x)

    return t2


def map_contours2(s1, s2, t1, t2):
    """
    Solves a variational problem for mapping positions along contour s1
    to positions along contour s2.

    Parameters
    ----------
    s1: tuple
        spline tuple as returned by splprep
    s2: tuple
        spline tuple as returned by splprep
    t1: 1d array
        list of spline parameters along s1
        defining positions centered on windows
    t2: 1d array
        list of spline parameters along s2
        as initial guesses

    Returns
    -------
    t2: 1d array
        updated positions along s2 minimizing the functional

    """

    N = len(t1)

    # The weight that balances the two terms (displacement and strain)
    # of the functional
    w = np.sum((np.concatenate(splev(t2, s2)) - np.concatenate(splev(t1, s1))) ** 2) / (
        N - 1
    )

    # The functional
    functional = Functional2(s1, s2, t1, w)

    # Change-of-basis matrix for specifying the linear constraints
    # ensure that spline parameters are monotonicialy growing (no crossings)
    A = np.zeros((N, N))
    for n in range(0, N):
        A[n, n] = 1
    A[0, N - 1] = -1
    for n in range(1, N):
        A[n, n - 1] = -1

    # Lower and upper bounds for the constraints
    lb = np.zeros((N,))
    lb[0] = -1
    ub = np.inf * np.ones((N,))

    # Minimization of the functional; we use high tolerances to
    # get the results faster
    try:
        result = minimize(
            fun=functional.f,
            x0=t2,
            method="trust-constr",
            constraints=LinearConstraint(A, lb, ub, keep_feasible=True),
            options={"gtol": 1e-2, "xtol": 1e-2},
        )
        # result = minimize(functional.f, t2, method='trust-constr', options={'gtol': 1e-12, 'xtol': 1e-12, 'barrier_tol': 1e-12})
        # result = minimize(functional.f, t2)
        t2 = result.x
        return t2
    except Exception:
        return None


def map_contours3(s1, s2, t1, t2):
    N = len(t1)
    functional = Functional3(s1, s2, t1, 0)
    result = least_squares(functional.f, t2, method="lm")
    t2 = result.x
    return t2


def show_edge_scatter(N, s1, s2, t1, t2, d, dmax=None):
    """Draw the cell-edge contour and the displacement vectors.
    The contour is drawn using a scatter plot to color-code the displacements."""

    # Evaluate splines at window locations and on fine-resolution grid
    c1 = splevper(t1, s1)
    c2 = splevper(t2, s2)
    c1p = splev(np.linspace(0, 1, N + 1), s1)
    c2p = splev(np.linspace(0, 1, N + 1), s2)

    # Interpolate displacements
    # d = 0.5 + 0.5 * d / np.max(np.abs(d))
    if len(d) < N + 1:
        d = np.interp(np.linspace(0, 1, N + 1), t1, d, period=1)
    if dmax is None:
        dmax = np.max(np.abs(d))
        if dmax == 0:
            dmax = 1

    # Plot results
    # matplotlib.use('PDF')
    lw = 1
    s = 1  # Scaling factor for the vectors

    plt.plot(c1p[0], c1p[1], "b", zorder=50, lw=lw)
    plt.plot(c2p[0], c2p[1], "r", zorder=100, lw=lw)
    # plt.scatter(c1p[0], c1p[1], c=d, cmap='bwr', vmin=-dmax, vmax=dmax, zorder=50, s1=lw)
    # # plt.colorbar(label='Displacement [pixels]')
    for j in range(len(t2)):
        plt.arrow(
            c1[0][j],
            c1[1][j],
            s * (c2[0][j] - c1[0][j]),
            s * (c2[1][j] - c1[1][j]),
            color="y",
            zorder=200,
            lw=lw,
        )
    # plt.arrow(c1[0][j], c1[1][j], s1 * u[0][j], s1 * u[1][j], color='y', zorder=200, lw=lw) # Show normal to curve
    plt.arrow(
        c1[0][0],
        c1[1][0],
        s * (c2[0][0] - c1[0][0]),
        s * (c2[1][0] - c1[1][0]),
        color="c",
        zorder=400,
        lw=lw,
    )


def show_edge_scatter_init(N, p, s1, s2, t1, t2, d, dmax=None):
    """Draw the cell-edge contour and the displacement vectors.
    The contour is drawn using a scatter plot to color-code the displacements."""

    # Evaluate splines at window locations and on fine-resolution grid
    c1 = splevper(t1, s1)
    c2 = splevper(t2, s2)
    c1p = splev(np.linspace(0, 1, N + 1), s1)
    c2p = splev(np.linspace(0, 1, N + 1), s2)

    # Interpolate displacements
    # d = 0.5 + 0.5 * d / np.max(np.abs(d))
    if len(d) < N + 1:
        d = np.interp(np.linspace(0, 1, N + 1), t1, d, period=1)
    if dmax is None:
        dmax = np.max(np.abs(d))
        if dmax == 0:
            dmax = 1

    # Plot results
    # matplotlib.use('PDF')
    lw = 1
    s = 1  # Scaling factor for the vectors

    (p.p1,) = plt.plot(c1p[0], c1p[1], "b", zorder=50, lw=lw)
    (p.p2,) = plt.plot(c2p[0], c2p[1], "r", zorder=100, lw=lw)
    # plt.scatter(c1p[0], c1p[1], c=d, cmap='bwr', vmin=-dmax, vmax=dmax, zorder=50, s1=lw)
    # # plt.colorbar(label='Displacement [pixels]')
    p.a = []
    for j in range(len(t2)):
        p.a.append(
            plt.arrow(
                c1[0][j],
                c1[1][j],
                s * (c2[0][j] - c1[0][j]),
                s * (c2[1][j] - c1[1][j]),
                color="y",
                zorder=200,
                lw=lw,
            )
        )
    # plt.arrow(c1[0][j], c1[1][j], s1 * u[0][j], s1 * u[1][j], color='y', zorder=200, lw=lw) # Show normal to curve
    p.a.append(
        plt.arrow(
            c1[0][0],
            c1[1][0],
            s * (c2[0][0] - c1[0][0]),
            s * (c2[1][0] - c1[1][0]),
            color="c",
            zorder=400,
            lw=lw,
        )
    )
    return p


def show_edge_scatter_update(N, p, s1, s2, t1, t2, d, dmax=None):
    """Draw the cell-edge contour and the displacement vectors.
    The contour is drawn using a scatter plot to color-code the displacements."""

    # Evaluate splines at window locations and on fine-resolution grid
    c1 = splevper(t1, s1)
    c2 = splevper(t2, s2)
    c1p = splev(np.linspace(0, 1, N + 1), s1)
    c2p = splev(np.linspace(0, 1, N + 1), s2)

    # Interpolate displacements
    # d = 0.5 + 0.5 * d / np.max(np.abs(d))
    if len(d) < N + 1:
        d = np.interp(np.linspace(0, 1, N + 1), t1, d, period=1)
    if dmax is None:
        dmax = np.max(np.abs(d))
        if dmax == 0:
            dmax = 1

    # Plot results
    # matplotlib.use('PDF')
    lw = 1
    s = 1  # Scaling factor for the vectors

    # p = Struct()
    # p.p1 = plt.plot(c1p[0], c1p[1], 'b', zorder=50, lw=lw)
    p.p1.set_data(c1p[0], c1p[1])
    # p.p2 = plt.plot(c2p[0], c2p[1], 'r', zorder=100, lw=lw)
    p.p2.set_data(c2p[0], c2p[1])
    # plt.scatter(c1p[0], c1p[1], c=d, cmap='bwr', vmin=-dmax, vmax=dmax, zorder=50, s1=lw)
    # # plt.colorbar(label='Displacement [pixels]')
    for a in p.a:
        a.remove()
    p.a = []
    for j in range(len(t2)):
        p.a.append(
            plt.arrow(
                c1[0][j],
                c1[1][j],
                s * (c2[0][j] - c1[0][j]),
                s * (c2[1][j] - c1[1][j]),
                color="y",
                zorder=200,
                lw=lw,
            )
        )
    # plt.arrow(c1[0][j], c1[1][j], s1 * u[0][j], s1 * u[1][j], color='y', zorder=200, lw=lw) # Show normal to curve
    p.a.append(
        plt.arrow(
            c1[0][0],
            c1[1][0],
            s * (c2[0][0] - c1[0][0]),
            s * (c2[1][0] - c1[1][0]),
            color="c",
            zorder=400,
            lw=lw,
        )
    )
    return p


def show_edge_line_aux(N, s, color, lw):
    c = splev(np.linspace(0, 1, N + 1), s)
    plt.plot(c[0], c[1], color=color, zorder=50, lw=lw)


def show_edge_line(N, s):
    """Draw the cell-edge contour using a colored line."""

    # Evaluate splines at window locations and on fine-resolution grid
    K = len(s)
    cmap = plt.cm.get_cmap("jet")
    lw = 0.1
    for k in range(K):
        show_edge_line_aux(N, s[k], cmap(k / (K - 1)), lw)
    plt.gcf().colorbar(
        plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=K - 1), cmap=cmap),
        label="Frame index",
    )


def compute_edge_image(N, shape, s, t, d, thickness, dmax=None):
    """Compute edge image

    Parameters
    ----------
    N: int
        number of interpolation points
    shape: tuple
        size of image
    s: spline object
        spline object as returned by splprep
    t: array
        vector of interpolation parameters
    d: array
        variable to use for contour coloring
    thickness: float
        thickness of contour
    dmax: float
        maximum of coloring variable

    Returns
    -------
    c: 2d array
        rasterized image
    """
    # create rasterized image
    c = rasterize_curve(N, shape, s, 0)
    mask = -1 < c
    # assign color along contour by interpolating the variable d
    c[mask] = np.interp(c[mask], t, d, period=1)
    c[np.logical_not(mask)] = 0
    # enlarge contour
    if thickness > 1:
        f = np.ones((thickness, thickness))
        n = convolve2d(mask, f, mode="same")
        mask = 0 < n
        n[np.logical_not(mask)] = 1
        c = convolve2d(c, f, mode="same") / n
    cmap = plt.cm.get_cmap("seismic")  # 'bwr'
    if dmax is None:
        dmax = np.max(np.abs(c))
    if dmax == 0:
        dmax = 1
    c = cmap(0.5 + 0.5 * c / dmax)[:, :, 0:3]
    c = (255 * c).astype(np.uint8)
    c *= np.stack((mask, mask, mask), -1)
    # plt.figure()
    # plt.imshow(c)
    # plt.get_current_fig_manager().window.showMaximized()
    # plt.show()
    return c


# def rasterize_curve(shape, s1, deltat):
#     """ Construct a mapping from edge pixels to spline arguments. """
#     delta = np.inf * np.ones(shape)
#     tau = - np.ones(shape)
#     t = np.mod(np.linspace(0, 1, 10001) - deltat, 1)
#     p1 = np.asarray(splev(t, s1))
#     pi = np.round(p1).astype(dtype=np.int)
#     for n in range(10001):
#         d0 = np.linalg.norm(p1[:, n]-pi[:, n])
#         if d0 < delta[pi[1, n], pi[0, n]]:
#             delta[pi[1, n], pi[0, n]] = d0
#             tau[pi[1, n], pi[0, n]] = t[n]
#     return tau


def rasterize_curve(N, shape, s, deltat):
    """
    Represent a contour as a grayscale image.
    If a pixel is part of the contour, then its intensity
    is equal to the parameter t of the closest point on the contour s(t).
    Otherwise it is equal to -1.

    Parameters
    ----------
    shape: tuple
        size of the desired image.
    s1: tuple
        spline tuple as returned by splprep
    deltat: float
        origin shift of the spline.

    Returns
    -------
    tau: 2d array
        rasterized image of the contour

    """

    delta = np.inf * np.ones(
        shape
    )  # Will store the distance between edge pixels and the closest points on the contour
    tau = -np.ones(
        shape
    )  # Will store the parameters t of the closest points on the contour; pixels that are not part of the contour will take the value -1
    t = np.linspace(0, 1, N + 1)  # The parameters of the points on the curve
    p = np.asarray(splev(t, s))  # The points on the curve
    t = np.mod(
        t - deltat, 1
    )  # Adjust the origin of the curve and account for periodicity of the parameterization
    pi = np.round(p).astype(
        dtype=np.int
    )  # Coordinates of the pixels that are part of the contour
    d0 = np.linalg.norm(
        p - pi, axis=0
    )  # Distances between the points on the contour and the nearest pixels
    for n in range(N + 1):  # For each point p[:, n] on the contour...
        if (
            d0[n] < delta[pi[1, n], pi[0, n]]
        ):  # ... if the distance to the nearest pixel is the smallest so far...
            delta[pi[1, n], pi[0, n]] = d0[n]  # ... remember this distance...
            tau[pi[1, n], pi[0, n]] = t[
                n
            ]  # ... and store the parameter t corresponding to p[:, n]
    return tau


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
    cvec = np.asarray(find_contours(mask, 0, fully_connected="high")[0], dtype=np.int)

    # Adjust the origin of the contour:
    # on the discrete contour cvec, find the closest point to the origin,
    # then apply a circular shift to cvec to make this point the first one.
    n0 = np.argmin(np.linalg.norm(cvec - origin, axis=1))
    cvec = np.roll(cvec, -n0, axis=0)

    # Compute the discrete arc length along the contour
    Lvec = compute_discrete_arc_length(cvec)

    # Compute the index of the mid-point for each window
    # Note that the arc length is being used as a coordinate along the curve
    n = np.zeros((I,), dtype=np.int)
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
    m = np.zeros((I,), dtype=np.int)
    for i in range(I):
        m[i] = np.argmin(np.linalg.norm(np.transpose(c) - np.flip(cvec[n[i]]), axis=1))
        c = [c[0][m[i]+1::], c[1][m[i]+1::]]
    m = m+1
    m[0] = 0
    m = np.cumsum(m)
    t_sel = t[m]

    return cvec_sel, t_sel
