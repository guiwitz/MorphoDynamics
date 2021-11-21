import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as ipw
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import display

from .show_plots import show_edge_vectorial_aux


class EdgeVectorialSlow:
    """
    Interactive UI to visualize the edge displacement over time"""

    def __init__(self, param, data, res):
        self.param = param
        self.data = data
        self.res = res
        self.mode = "displacement"

    def create_interface(self):
        out = ipw.Output()

        mode_selector = ipw.RadioButtons(
            options=["displacement", "curvature"],
            value="displacement",
            description="Mode:",
        )

        def mode_change(change):
            with out:
                self.set_mode(change["new"])
                plt.figure(self.fig.number)
                show_edge_vectorial_aux(
                    self.param,
                    self.data,
                    self.res,
                    time_slider.get_state()["value"],
                    curvature=self.curvature,
                    fig_ax=(self.fig, self.ax),
                )
                self.fig.canvas.draw_idle()

        mode_selector.observe(mode_change, names="value")

        time_slider = ipw.IntSlider(
            description="Time",
            value=1,
            min=1,
            max=self.data.K - 2,
            continuous_update=False,
            layout=ipw.Layout(width="100%"),
        )

        def time_change(change):
            with out:
                plt.figure(self.fig.number)
                show_edge_vectorial_aux(
                    self.param,
                    self.data,
                    self.res,
                    change["new"],
                    curvature=self.curvature,
                    fig_ax=(self.fig, self.ax),
                )
                self.fig.canvas.draw_idle()

        time_slider.observe(time_change, names="value")

        self.set_mode(mode_selector.get_state("value"))
        with out:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            show_edge_vectorial_aux(
                self.param,
                self.data,
                self.res,
                1,
                curvature=False,
                fig_ax=(self.fig, self.ax),
            )
            display(self.fig.canvas)
        self.interface = ipw.VBox([time_slider, out])

    def set_mode(self, mode):
        self.curvature = mode == "curvature"
