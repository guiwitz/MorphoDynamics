import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as ipw
from matplotlib.backends.backend_pdf import PdfPages

from ..displacementestimation import compute_curvature


class Curvature:
    def __init__(self, param, data, res):
        self.param = param
        self.data = data
        self.res = res

        self.curvature = np.zeros((self.param.n_curve, self.data.K))
        for k in range(self.data.K):
            self.curvature[:, k] = compute_curvature(
                self.res.spline[k],
                np.linspace(0, 1, self.param.n_curve, endpoint=False),
            )

    def create_interface(self):
        out = ipw.Output()

        cmax = np.max(np.abs(self.curvature))

        range_slider = ipw.FloatSlider(
            value=cmax,
            description="Range:",
            min=0,
            max=cmax,
            step=0.1,
            continuous_update=False,
            layout=ipw.Layout(width="100%"),
        )

        def range_change(change):
            with out:
                plt.figure(self.fig.number)
                self.show_curvature(change["new"])

        range_slider.observe(range_change, names="value")

        with out:
            self.fig = plt.figure(figsize=(8, 6))
            self.show_curvature(range_slider.value)

        self.interface = ipw.VBox([range_slider, out])

    def show_curvature(self, cmax, export=True):
        plt.clf()
        plt.gca().set_title("Curvature")
        if export:
            pp = PdfPages(os.path.join(self.param.resultdir, "Curvature.pdf"))
        plt.imshow(self.curvature, cmap="seismic", vmin=-cmax, vmax=cmax)
        plt.colorbar(label="Curvature")
        plt.axis("auto")
        plt.xlabel("Frame index")
        plt.ylabel("Position on contour")
        # imsave(param.resultdir + 'Curvature.tif', curvature.astype(np.float32), compress=6)
        if export:
            pp.savefig()
            pp.close()
