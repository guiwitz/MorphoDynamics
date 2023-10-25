import numpy as np
import matplotlib.pyplot as plt


def correlate(x, y, normalization=None, removemean=True):
    def npcorrelate(u, v):
        return np.correlate(u, v, mode="full")

    if removemean:
        x = x - np.mean(x)
        y = y - np.mean(y)
    c = npcorrelate(x, y)
    if normalization == "unbiased":
        c /= npcorrelate(np.ones(x.shape), np.ones(y.shape))
    elif normalization == "Pearson":
        if np.linalg.norm(x) * np.linalg.norm(y) == 0:
            print("alert")
        c /= np.linalg.norm(x) * np.linalg.norm(y)
    elif normalization == "Pearson-unbiased":
        e = np.sqrt(
            npcorrelate(x ** 2, np.ones(y.shape))
            * npcorrelate(np.ones(x.shape), y ** 2)
        )
        e[e == 0] = 1
        c /= e
    return c


def test_correlate():
    K = 100
    # Visualize position corresponding to zero lag
    x = np.zeros((K,))
    x[0] = 1
    y = np.zeros((K,))
    y[0] = 1
    plt.figure()
    plt.plot(correlate(x, y))
    # Verify that correlation of ones gives normalization coefficients for unbiased mode
    x = np.ones((K,))
    y = np.ones((K,))
    plt.figure()
    plt.plot(correlate(x, y, removemean=False))
    # Verify autocorrelation of a sine with Pearson-unbiased normalization
    x = np.sin(2 * np.pi * np.array(range(K)) / 10)
    plt.figure()
    plt.plot(correlate(x, x, normalization="Pearson-unbiased"))
    plt.show()
    quit()


def correlate_arrays(x, y, normalization):
    I = x.shape[0]
    c = np.zeros((I, x.shape[1] + y.shape[1] - 1))
    for i in range(I):
        c[i] = correlate(x[i], y[i], normalization=normalization, removemean=True)
    return c


def get_extent(A, B, I):
    A, B = max(A, B), min(A, B)
    return -B + 1 - 0.5, -B + 1 + A + B - 2 + 0.5, I - 1 + 0.5, -0.5


def get_range(A, B):
    A, B = max(A, B), min(A, B)
    return range(-B + 1, -B + 1 + A + B - 1)


def show_average_correlation(fh, c, x, y):
    fh.open_figure("Cross-correlation", 1, (16, 9))
    t = get_range(x.shape[1], y.shape[1])
    cmean = np.mean(c, axis=0)
    plt.plot(t, cmean)
    # # 95% confidence intervals for the Pearson correlation
    # plt.plot([t[0], t[-1]], 1.96/math.sqrt(len(t))*np.array([1, 1]), 'c')
    # plt.plot([t[0], t[-1]], -1.96/math.sqrt(len(t))*np.array([1, 1]), 'c')
    plt.ylim(-np.max(cmean), np.max(cmean))
    plt.grid()
    fh.show()
