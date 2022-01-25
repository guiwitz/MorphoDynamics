from scipy.interpolate import splev
import numpy as np
import math
from .splineutils import splevper


class Functional:
    def __init__(self, tck1, tck2, p, w):
        self.tck1 = tck1
        self.tck2 = tck2
        self.p = p
        self.w = w

    def transform(self, o):
        """ Change of variable that is necessary to express the constraints of the optimization problem as simple inequalities. """
        n = len(o)
        r = np.zeros((n + 1,))
        r[0] = o[0]
        for i in range(1, n):
            r[i] = o[i] - o[i - 1]
        r[n] = o[0] - o[n - 1]
        return r

    def inversetransform(self, r):
        """ Inverse of the change of variable. """
        n = len(r) - 1
        o = np.zeros((n,))
        o[0] = r[0]
        for i in range(1, n):
            o[i] = r[i] + o[i - 1]
        return o

    def f(self, r):
        """The functional to be minimized, adapted from the following paper:
        Ma Dagliyan Hahn Danuser - Profiling cellular morphodynamics by spatiotemporal spectrum decomposition
        Note that here periodic boundary conditions are enforced."""
        o = self.inversetransform(r)
        n = len(o)
        f = np.zeros((3 * n,))
        f[0 : 2 * n] = np.concatenate(splev(np.mod(o, 1), self.tck2)) - np.concatenate(
            splev(np.mod(self.p, 1), self.tck1)
        )
        for i in range(1, n):
            f[i + 2 * n - 1] = (
                math.sqrt(self.w) * (o[i] - o[i - 1]) / (self.p[i] - self.p[i - 1])
            )
        # f[i + 2 * n - 1] = math.sqrt(self.w) * (self.p1[i] - self.p1[i - 1]) / (o[i] - o[i - 1])
        f[3 * n - 1] = (
            math.sqrt(self.w) * (1 + o[0] - o[n - 1]) / (1 + self.p[0] - self.p[n - 1])
        )
        # f[3*n-1] = math.sqrt(self.w) * (1+self.p1[0]-self.p1[n-1]) / (1+o[0]-o[n-1])
        return f


class Functional2:
    def __init__(self, tck1, tck2, p, w):
        self.tck1 = tck1
        self.tck2 = tck2
        self.p = p
        self.w = w

    # def transform(self, o):
    #     """ Change of variable that is necessary to express the constraints of the optimization problem as simple inequalities. """
    #     n = len(o)
    #     r = np.zeros((n + 1,))
    #     r[0] = o[0]
    #     for i in range(1, n):
    #         r[i] = o[i] - o[i - 1]
    #     r[n] = o[0] - o[n - 1]
    #     return r
    #
    # def inversetransform(self, r):
    #     """ Inverse of the change of variable. """
    #     n = len(r) - 1
    #     o = np.zeros((n,))
    #     o[0] = r[0]
    #     for i in range(1, n):
    #         o[i] = r[i] + o[i - 1]
    #     return o

    def f(self, q):
        """The functional to be minimized, adapted from the following paper:
        Ma Dagliyan Hahn Danuser - Profiling cellular morphodynamics by spatiotemporal spectrum decomposition
        Note that here periodic boundary conditions are enforced."""
        n = len(q)
        f = np.zeros((3 * n,))
        f[0 : 2 * n] = np.concatenate(splevper(q, self.tck2)) - np.concatenate(
            splevper(self.p, self.tck1)
        )
        for i in range(1, n):
            f[i + 2 * n - 1] = (
                math.sqrt(self.w) * (q[i] - q[i - 1]) / (self.p[i] - self.p[i - 1])
            )
        # f[i + 2 * n - 1] = math.sqrt(self.w) * (self.p1[i] - self.p1[i - 1]) / (o[i] - o[i - 1])
        f[3 * n - 1] = (
            math.sqrt(self.w) * (1 + q[0] - q[n - 1]) / (1 + self.p[0] - self.p[n - 1])
        )
        # f[3*n-1] = math.sqrt(self.w) * (1+self.p1[0]-self.p1[n-1]) / (1+o[0]-o[n-1])
        return np.sum(f ** 2)


class Functional3:
    def __init__(self, tck1, tck2, p, w):
        self.tck1 = tck1
        self.tck2 = tck2
        self.p = p
        self.w = w

    def f(self, q):
        """The functional to be minimized, adapted from the following paper:
        Ma Dagliyan Hahn Danuser - Profiling cellular morphodynamics by spatiotemporal spectrum decomposition
        Note that here periodic boundary conditions are enforced."""
        n = len(q)
        f = np.zeros((3 * n,))
        f[0 : 2 * n] = np.concatenate(splevper(q, self.tck2)) - np.concatenate(
            splevper(self.p, self.tck1)
        )
        for i in range(1, n):
            f[i + 2 * n - 1] = (
                math.sqrt(self.w) * (q[i] - q[i - 1]) / (self.p[i] - self.p[i - 1])
            )
        # f[i + 2 * n - 1] = math.sqrt(self.w) * (self.p1[i] - self.p1[i - 1]) / (o[i] - o[i - 1])
        f[3 * n - 1] = (
            math.sqrt(self.w) * (1 + q[0] - q[n - 1]) / (1 + self.p[0] - self.p[n - 1])
        )
        # f[3*n-1] = math.sqrt(self.w) * (1+self.p1[0]-self.p1[n-1]) / (1+o[0]-o[n-1])
        return f
