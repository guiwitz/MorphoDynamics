import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import ipywidgets as ipw

from .show_plots import show_edge_scatter
from ..splineutils import edge_colored_by_displacement
from ..splineutils import (
    spline_curvature)

def animate_edge_vect(data, param, res, fig, ax, k, curvature=False):
    
    ax.set_title(f'Frame {k-1} to {k}')
    image = data.load_frame_morpho(k)
    for a in ax.get_children():
        if isinstance(a, FancyArrow):
            a.remove()

    if curvature:
        f = spline_curvature(res.spline[k], np.linspace(0, 1, param.n_curve + 1))
    else:
        f = res.displacement[:, k]

    ax.get_images()[0].set_data(image)

    ax.lines.pop(0)
    ax.lines.pop(0)

    fig, ax = show_edge_scatter(
        param.n_curve,
        res.spline[k - 1],  # res.spline[k],
        res.spline[k],  # res.spline[k + 1],
        res.param0[k],
        res.param[k],
        f,
        fig_ax=(fig, ax),
    )

def interact_edge_vect(data, param, res, fig, ax):
    int_box = ipw.interactive(lambda k: animate_edge_vect(data, param, res, fig, ax, k), k=ipw.IntSlider(1, min=1, max=data.K-2))
    return int_box

def animate_edge_scatter_coloured_by_displacement(param, data, res, k, N, fig, ax, width=1):
    im_disp = edge_colored_by_displacement(data, res, t=k, N=N, enlarge_width=width)
    ax.get_images()[0].set_data(im_disp)

def interact_edge_scatter_coloured_by_displacement(data, param, res, N, fig, ax, width=1):
    int_box = ipw.interactive(lambda k: animate_edge_scatter_coloured_by_displacement(
        param, data, res, k, N, fig, ax, width), k=ipw.IntSlider(1, min=1, max=data.K-2))
    return int_box


