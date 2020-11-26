import os
import matplotlib.pyplot as plt
import ipywidgets as ipw
from matplotlib.backends.backend_pdf import PdfPages

from .show_plots import show_signals_aux


def show_signals(param, data, res, mode, size=(16, 9), export=True):
    f = plt.figure(figsize=size)

    if export:
        pp = PdfPages(os.path.join(param.resultdir, "Signal " + mode + ".pdf"))
    for m in range(len(data.signalfile)):
        for j in range(res.mean.shape[1]):
            plt.figure(f.number)
            show_signals_aux(param, data, res, m, j, mode)
            if export:
                pp.savefig()
    if export:
        pp.close()


class Signals:
    def __init__(self, param, data, res, mode=None):
        self.param = param
        self.data = data
        self.res = res

    def get_signal(self, name):
        for m in range(len(self.data.signalfile)):
            if name == self.data.get_channel_name(m):
                return m

    def create_interface(self):
        out = ipw.Output()

        signal_selector = ipw.RadioButtons(
            options=self.data.signal_name,
            value=self.data.signal_name[0],
            description="Signal:",
        )

        def channel_change(change):
            # plt.figure(self.fig.number)
            with out:
                self.fig, self.ax = show_signals_aux(
                    self.param,
                    self.data,
                    self.res,
                    self.get_signal(change["new"]),
                    layer_text.value,
                    mode_selector.value,
                    fig_ax=(self.fig, self.ax),
                )

        signal_selector.observe(channel_change, names="value")

        layer_text = ipw.BoundedIntText(
            value=0, min=0, max=self.res.J - 1, description="Layer:"
        )

        def layer_change(change):
            with out:
                # plt.figure(self.fig.number)
                self.fig, self.ax = show_signals_aux(
                    self.param,
                    self.data,
                    self.res,
                    self.get_signal(signal_selector.value),
                    layer_text.value,
                    mode_selector.value,
                    fig_ax=(self.fig, self.ax),
                )

        layer_text.observe(layer_change, names="value")

        mode_selector = ipw.RadioButtons(
            options=["Mean", "Variance"], value="Mean", description="Mode:"
        )

        def mode_change(change):
            with out:
                # plt.figure(self.fig.number)
                self.fig, self.ax = show_signals_aux(
                    self.param,
                    self.data,
                    self.res,
                    self.get_signal(signal_selector.value),
                    layer_text.value,
                    mode_selector.value,
                    fig_ax=(self.fig, self.ax),
                )

        mode_selector.observe(mode_change, names="value")

        with out:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.fig, self.ax = show_signals_aux(
                self.param,
                self.data,
                self.res,
                self.get_signal(signal_selector.value),
                layer_text.value,
                mode_selector.value,
                fig_ax=(self.fig, self.ax),
            )

        self.interface = ipw.VBox([signal_selector, layer_text, mode_selector, out])
