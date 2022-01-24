import math
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.colors import Normalize
import ipywidgets as ipw
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import splev

from .. import splineutils

out = ipw.Output()


def show_circularity(data, res, size=(16, 9)):
    """
    Display length, area and circularity information for time-lapse.

    Parameters
    ----------
    data: data object
        created from dataset.Data
    res: res object
        created from results.Results
    size: tuple
        image size

    Returns
    -------
    fig: matplotlib figure
    ax: matplotlib axis

    """

    length = np.zeros((data.K,))
    area = np.zeros((data.K,))
    for k in range(data.K):
        length[k] = splineutils.spline_contour_length(res.spline[k])
        area[k] = splineutils.spline_area(res.spline[k])

    fig, ax = plt.subplots(1, 3, figsize=size)
    ax[0].plot(length)
    ax[0].set_title("Length")

    ax[1].plot(area)
    ax[1].set_title("Area")

    ax[2].plot(length ** 2 / area / 4 / math.pi)
    ax[2].set_title("Circularity = Length^2 / Area / 4 / pi")

    fig.tight_layout()

    return fig, ax


def show_edge_line_aux(N, s, color, lw, fig_ax=None):
    """
    Plot as spline s of color color by interpolating N points.

    Parameters
    ----------
    N: int
        number of interpolation points
    s: spline object
        as returned by splprep
    color: matplotlib color
    lw: curve thickness
    fig_ax: tuple
        matplotlib figure and axes

    Returns
    -------
    fig: matplotlib figure
    ax: matplotlib axis

    """

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    c = splev(np.linspace(0, 1, N + 1), s)
    ax.plot(c[0], c[1], color=color, zorder=50, lw=lw)

    return fig, ax


def show_edge_line(N, s, lw=0.1, fig_ax=None):
    """
    Draw the cell-edge contour of all time points
    using a colored line.

    Parameters
    ----------
    N: int
        number of interpolation points
    s: spline object
        as returned by splprep
    lw: curve thickness
    fig_ax: tuple
        matplotlib figure and axes

    Returns
    -------
    fig: matplotlib figure
    ax: matplotlib axis

    """

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    # Evaluate splines at window locations and on fine-resolution grid
    K = len(s)
    cmap = plt.cm.get_cmap("jet")

    for k in range(K):
        fig, ax = show_edge_line_aux(N, s[k], cmap(k / (K - 1)), lw, fig_ax=(fig, ax))
    fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=K - 1), cmap=cmap),
        label="Frame index",
    )
    return fig, ax


def show_edge_overview(param, data, res, lw=0.1, size=(12, 9), fig_ax=None):
    """
    Display image of first time point and all contour splines
    overlayed on top.

    Parameters
    ----------
    param: param object
        created from parameters.Param
    data: data object
        created from dataset.Data
    res: res object
        created from results.Results
    lw: float
        spline curves thickness
    size: tuple
        image size
    fig_ax: tuple
        matplotlib figure and axes

    Returns
    -------
    fig: matplotlib figure
    ax: matplotlib axis

    """

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=size)
    else:
        fig, ax = fig_ax

    ax.set_title("Edge overview")
    ax.imshow(data.load_frame_morpho(0), cmap="gray")
    fig, ax = show_edge_line(param.n_curve, res.spline, lw, (fig, ax))
    fig.tight_layout()

    return fig, ax


def show_edge_vectorial_aux(param, data, res, k, curvature=False, fig_ax=None):
    """
    Plot time point k with the contour and the displacement vectors
    overlayed. The contour is color-coded to represent either
    displacement or curvature.

    Parameters
    ----------
    param: param object
        created from parameters.Param
    data: data object
        created from dataset.Data
    res: res object
        created from results.Results
    k: int
        time point
    curvature: bool
        represent curvature instead of displacement

    Returns
    -------
    fig: matplotlib figure
    ax: matplotlib axis

    """

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
        ax.clear()
        plt.figure(fig.number)

    # plt.clf()
    ax.set_title("Frame " + str(k) + " to frame " + str(k + 1))
    ax.imshow(data.load_frame_morpho(k), cmap="gray")

    #N =  param.n_curve + 1
    if curvature:
        N = 3 * len(res.spline[k][0])
        f = splineutils.spline_curvature(res.spline[k], np.linspace(0, 1, N))
    else:
        f = res.displacement[:, k]

    fig, ax = show_edge_scatter(
        param.n_curve,
        res.spline[k - 1],  # res.spline[k],
        res.spline[k],  # res.spline[k + 1],
        res.param0[k],
        res.param[k],
        f,
        fig_ax=(fig, ax),
    )  # Show edge structures (spline curves, displacement vectors/curvature)
    plt.tight_layout()
    return fig, ax


def save_edge_vectorial_movie(param, data, res, curvature=False, size=(12, 9)):
    if curvature:
        name = "Edge_animation_curvature"
    else:
        name = "Edge_animation_displacement"

    with out:
        fig, ax = plt.subplots(figsize=size)
        writer = imageio.get_writer(os.path.join(param.analysis_folder, name + ".gif"))

        for k in range(data.K - 1):
            fig, ax = show_edge_vectorial_aux(
                param, data, res, k, curvature, fig_ax=(fig, ax)
            )
            fig.savefig(os.path.join(param.analysis_folder, "temp.png"))
            writer.append_data(
                imageio.imread(os.path.join(param.analysis_folder, "temp.png"))
            )
        writer.close()
        plt.close(fig)


def show_edge_scatter(N, s1, s2, t1, t2, d, dmax=None, fig_ax=None):
    """Draw the cell-edge contour and the displacement vectors.
    The contour is drawn using a scatter plot to color-code the displacements."""

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
        plt.figure(fig.number)

    # Evaluate splines at window locations and on fine-resolution grid
    c1 = splineutils.splevper(t1, s1)
    c2 = splineutils.splevper(t2, s2)
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

    ax.plot(c1p[0], c1p[1], "b", zorder=50, lw=lw)
    ax.plot(c2p[0], c2p[1], "r", zorder=100, lw=lw)
    # plt.scatter(c1p[0], c1p[1], c=d, cmap='bwr', vmin=-dmax, vmax=dmax, zorder=50, s1=lw)
    # # plt.colorbar(label='Displacement [pixels]')
    for j in range(len(t2)):
        ax.arrow(
            c1[0][j],
            c1[1][j],
            s * (c2[0][j] - c1[0][j]),
            s * (c2[1][j] - c1[1][j]),
            color="y",
            zorder=200,
            lw=lw,
        )
    # plt.arrow(c1[0][j], c1[1][j], s1 * u[0][j], s1 * u[1][j], color='y', zorder=200, lw=lw) # Show normal to curve
    ax.arrow(
        c1[0][0],
        c1[1][0],
        s * (c2[0][0] - c1[0][0]),
        s * (c2[1][0] - c1[1][0]),
        color="c",
        zorder=400,
        lw=lw,
    )
    return fig, ax

def show_edge_raster_coloured_by_feature(
    data, res, k, feature, N=None, width=1, fig_ax=None, normalize=False, cmap_name='seismic'):
    """Display the rasterized contour colored by a given feature on top of image.

    Parameters
    ----------
    data : data object
    res : result object
    k : int
        time point
    feature : str
        feature for coloring 'displacement', 'displacement_cumul', 'curvature'
    N : int
        number of points for contour generation, default None
    width : int, optional
        width of contour for display, by default 1
    fig_ax : tuple, optional
        matplotlib figure-axis tuple, by default None
    normalize : bool, optional
        normalize intensity over time-lapse, by default False
    cmap_name : str, optional
        matplotlib colormap, by default 'seismic'

    Returns
    -------
    fig, ax: Matplotlib figure and axis

    """
    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
        plt.figure(fig.number)

    im_disp, mask = splineutils.edge_colored_by_features(
        data, res, t=k, feature=feature, N=N, enlarge_width=width)
    min_val = None
    max_val = None
    if normalize:
        if feature == 'displacement':
            min_val = res.displacement.min()
            max_val = res.displacement.max()
        elif feature == 'displacement_cumul':
            min_val = np.cumsum(res.displacement, axis=1).min()
            max_val = np.cumsum(res.displacement, axis=1).max()
    
    im_disp_coloured = colorize_raster(
        im_disp, cmap_name=cmap_name, 
        min_val=min_val, max_val=max_val,
        mask=mask)

    ax.imshow(data.load_frame_morpho(k), cmap='gray')
    ax.imshow(im_disp_coloured)
    ax.set_title("Frame " + str(k))

    fig.tight_layout()
    return fig, ax

def colorize_raster(im, cmap_name, min_val=None, max_val=None, mask=None, alpha=0.5):
    """Colorize an image with a given colormap.

    Parameters
    ----------
    im : ndarray
        image to colorize
    cmap_name : str
        Matplotlib colormap
    min_val : float, optional
        min value to display, by default min of image
    max_val : [type], optional
        max value to display, by default max of image
    mask : ndarray, optional
        mask to make empty regions transparent, by default None
    alpha : float, optional
        transparency of image, by default 0.5

    Returns
    -------
    c: ndarray
        colorized image (nxmx4)
    """
    if mask is None:
        mask = np.ones(im.shape, dtype=np.bool8)
    if min_val is None:
        min_val = im.min()
    if max_val is None:
        max_val = im.max()
    cmap = plt.cm.get_cmap(cmap_name)  # 'bwr'
    c = cmap(0.5 + 0.5 * (im-min_val) / (max_val-min_val))
    c = (255 * c).astype(np.uint8)
    c[:,:,3] = int(255*alpha)
    c *= np.stack((mask, mask, mask, mask), -1)
    return c

def show_displacement(param, res, size=(16, 9), fig_ax=None):

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=size)
    else:
        fig, ax = fig_ax
        plt.figure(fig.number)

    ax.set_title("Displacement")
    im = ax.imshow(res.displacement, cmap="seismic")
    plt.axis("auto")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Window index")
    plt.colorbar(im, label="Displacement [pixels]")

    if hasattr(param, "scaling_disp"):
        cmax = param.scaling_disp
    else:
        cmax = np.max(np.abs(res.displacement))
    im.set_clim(-cmax, cmax)

    return fig, ax


def show_cumdisplacement(res, size=(16, 9), fig_ax=None):

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=size)
    else:
        fig, ax = fig_ax
        plt.figure(fig.number)

    dcum = np.cumsum(res.displacement, axis=1)

    ax.set_title("Cumulative displacement")
    im = ax.imshow(dcum, cmap="seismic")
    plt.axis("auto")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Window index")
    plt.colorbar(im, label="Displacement [pixels]")
    cmax = np.max(np.abs(dcum))
    im.set_clim(-cmax, cmax)

    return fig, ax


def show_signals_aux(
    data, res, signal_index, layer_index, mode, fig_ax=None, size=(16, 9), layer_title=False
):
    """
    Display window-kymograph of a signal.

    Parameters
    ----------
    data: data object
        created from dataset.Data
    res: res object
        created from results.Results
    signal_index: int
        signal index
    layer_index: int
        layer index
    mode; str
        "Mean" or "Variance"
    fig_ax: tuple
        matplotlib figure and axes
    size: tuple
        figure size
    layer_title: bool
        If true, add only layer as title

    Returns
    -------
    fig; matplotlib figure
    ax: matplotlib axis

    """

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=size)
    else:
        fig, ax = fig_ax
        ax.clear()
        plt.figure(fig.number)

    if mode == "Mean":
        f = res.mean[signal_index, layer_index, 0 : res.I[layer_index], :]
    elif mode == "Variance":
        f = res.var[signal_index, layer_index, 0 : res.I[layer_index], :]

    if layer_title:
        ax.set_title("Layer: " + str(layer_index))
    else:
        ax.set_title("Signal: " + data.get_channel_name(signal_index) + " - Layer: " + str(layer_index))

    im = ax.imshow(f, cmap="jet")
    if len(fig.axes) == 2:

        fig.axes[1].clear()
        fig.colorbar(im, cax=fig.axes[1], label=mode)

    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label=mode)
    plt.axis("auto")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Window index")
    ax.set_aspect("equal")

    return fig, ax


def save_signals(param, data, res, modes=None, size=(16, 9)):

    if not os.path.isdir(os.path.join(param.analysis_folder, "signals")):
        os.makedirs(os.path.join(param.analysis_folder, "signals"))

    if modes is None:
        modes = ["Mean", "Variance"]
    for mode in modes:
        for j in range(res.mean.shape[1]):

            with out:
                fig, ax = plt.subplots(len(data.signal_name), 1, figsize=(4, 4))
                if len(data.signal_name) == 1:
                    ax = np.array([ax])
                    ax = ax[np.newaxis, :]

                for m in range(len(data.signal_name)):

                    show_signals_aux(
                        param,
                        data,
                        res,
                        m,
                        j,
                        "Mean",
                        (fig, ax[m, 0]),
                        # layer_title=True,
                    )

            fig.savefig(
                os.path.join(
                    param.analysis_folder,
                    "signals",
                    "Signal_" + str(m) + "_" + mode + "_layer_" + str(j) + ".png",
                )
            )


def show_curvature(data, res, cmax=None, fig_ax=None):
    """Display curvature as a function of time

    Parameters
    ----------
    data : data object
    res : result object
    cmax : float, optional
        maximal curvature value to display, default None
    fig_ax : tuple, optional
        (fig, ax), by default None

    Returns
    -------
    fig : matplotlib figure
    ax : matplotlib axis

    """

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
        ax.clear()
        plt.figure(fig.number)

    N = 3 * int(np.max([splineutils.spline_contour_length(r) for r in res.spline]))
    #N = np.max([3*len(r[0]) for r in res.spline])
    curvature = np.zeros((N, data.K))
    for k in range(data.K):
        curvature[:, k] = splineutils.spline_curvature(
            res.spline[k],
            np.linspace(0, 1, N, endpoint=False),
        )
    if cmax is None:
        cmax = np.max(np.abs(curvature))

    ax.set_title("Curvature")

    im = ax.imshow(curvature, cmap="seismic", vmin=-cmax, vmax=cmax)
    plt.colorbar(im, label="Curvature", ax=ax)
    plt.axis("auto")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Position on contour")

    return fig, ax
