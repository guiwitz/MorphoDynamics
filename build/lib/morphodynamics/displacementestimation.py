from copy import deepcopy

from scipy.interpolate import splev
from scipy.optimize import least_squares, minimize, LinearConstraint
import numpy as np

from .functionaldefinition import Functional, Functional2, Functional3
from .splineutils import splevper


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
