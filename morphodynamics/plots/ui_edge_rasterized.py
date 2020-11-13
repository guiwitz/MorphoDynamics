import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as ipw
from tifffile import TiffWriter
import imageio

from ..displacementestimation import compute_curvature, compute_edge_image

# from .show_plots import show_edge_rasterized_aux


class EdgeRasterized:
    def __init__(self, param, data, res, mode=None):
        self.param = param
        self.data = data
        self.res = res

        self.y = np.zeros((self.data.K,) + self.data.shape)
        for k in range(self.data.K):
            self.y[k] = self.data.load_frame_morpho(k)
        self.y = 255 * self.y / np.max(self.y)

    def create_interface(self):
        out = ipw.Output()

        mode_selector = ipw.RadioButtons(
            options=[
                "border",
                "displacement",
                "cumulative displacement",
                "curvature",
            ],
            value="displacement",
            description="Mode:",
        )

        def mode_change(change):
            with out:
                self.set_mode(change["new"])
                self.set_normalization(normalization_selector.value)
                self.fig, self.ax = self.plot(
                    time_slider.value, (self.fig, self.ax)
                )

        mode_selector.observe(mode_change, names="value")

        normalization_selector = ipw.RadioButtons(
            options=["global", "frame-by-frame"],
            value="frame-by-frame",
            layout={"width": "max-content"},  # If the items' names are long
            description="Normalization:"
            # disabled=False
        )

        def normalization_change(change):
            with out:
                self.set_normalization(change["new"])
                self.fig, self.ax = self.plot(
                    time_slider.value, (self.fig, self.ax)
                )

        normalization_selector.observe(normalization_change, names="value")

        time_slider = ipw.IntSlider(
            description="Time",
            min=0,
            max=self.data.K - 2,
            continuous_update=False,
            layout=ipw.Layout(width="100%"),
        )

        def time_change(change):
            with out:
                self.fig, self.ax = self.plot(
                    change["new"], (self.fig, self.ax)
                )

        time_slider.observe(time_change, names="value")

        # display(mode_selector)
        # display(normalization_selector)
        # display(time_slider)

        self.set_mode(mode_selector.value)
        self.set_normalization(normalization_selector.value)

        with out:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.fig, self.ax = self.plot(0, (self.fig, self.ax))
        # display(out)
        self.interface = ipw.VBox(
            [mode_selector, normalization_selector, time_slider, out]
        )

    def set_mode(self, mode):
        self.mode = mode
        if mode == "border":
            self.d = np.ones(self.res.displacement.shape)
            self.name = "Edge animation (border)"
        elif mode == "displacement":
            self.d = self.res.displacement
            self.name = "Edge animation (displacement)"
        elif mode == "cumulative displacement":
            self.d = np.cumsum(self.res.displacement, axis=1)
            self.name = "Edge animation (cumulative displacement)"
        elif mode == "curvature":
            t = np.linspace(0, 1, self.param.n_curve, endpoint=False)
            self.d = np.zeros((self.param.n_curve, self.data.K))
            for k in range(self.data.K):
                self.d[:, k] = compute_curvature(self.res.spline[k], t)
            self.name = "Edge animation (curvature)"

    def set_normalization(self, normalization):
        if normalization == "global":
            self.dmax = np.max(np.abs(self.d))
        else:
            self.dmax = None

    def show_edge_rasterized_aux(self, d, dmax, k, mode):
        """
        Compute edge image.

        Parameters
        ----------
        param: param object
            created from parameters.Param
        data: data object
            created from dataset.Data
        res: res object
            created from results.Results
        d: array
            variable to represent along contour
        dmax: float
            max value of the variable
        k: int
            time point
        mode: str
            "curvature"

        """

        if mode == "curvature":
            x = compute_edge_image(
                self.param.n_curve,
                self.data.shape,
                self.res.spline[k],
                np.linspace(0, 1, self.param.n_curve, endpoint=False),
                d[:, k],
                3,
                dmax,
            )
        else:
            x = compute_edge_image(
                self.param.n_curve,
                self.data.shape,
                self.res.spline[k],
                self.res.param0[k],
                d[:, k],
                3,
                dmax,
            )

        return x

    def get_image(self, k):
        x = self.show_edge_rasterized_aux(
            self.d,
            self.dmax,
            k,
            self.mode,
        )
        y0 = np.stack((self.y[k], self.y[k], self.y[k]), axis=-1)
        x[x == 0] = y0[x == 0]
        return x

    def save(self):
        tw = TiffWriter(self.param.resultdir + self.name + ".tif")
        for k in range(self.data.K - 1):
            tw.save(self.get_image(k), compress=6)
        tw.close()

    def plot(self, k, fig_ax=None):

        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax
            plt.figure(fig.number)
            ax.clear()
        ax.set_title("Frame " + str(k))
        ax.imshow(self.get_image(k))
        plt.tight_layout()
        return fig, ax

    def save_movie(self, mode):

        name = "rasterized_" + mode
        self.set_mode(mode)
        self.set_normalization("global")
        out = ipw.Output()
        with out:
            fig, ax = plt.subplots()
            writer = imageio.get_writer(
                os.path.join(self.param.resultdir, name + ".gif")
            )
            for k in range(self.data.K - 1):
                fig, ax = self.plot(k, (fig, ax))
                fig.savefig(os.path.join(self.param.resultdir, "temp.png"))
                writer.append_data(
                    imageio.imread(
                        os.path.join(self.param.resultdir, "temp.png")
                    )
                )
        writer.close()
        plt.close(fig)
