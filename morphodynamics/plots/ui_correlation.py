import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as ipw
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

from ..correlation import show_correlation_core, correlate_arrays, get_range


class Correlation:
    def __init__(self, param, data, res, mode=None):
        self.param = param
        self.data = data
        self.res = res

        self.out = ipw.Output()
        self.out_avg = ipw.Output()
        self.out_compl = ipw.Output()

    def create_interface(self):

        with self.out:
            # main correlation plot
            self.fig, self.ax = plt.subplots(figsize=(4, 4))
        with self.out_avg:
            # average plot
            self.fig_avg, self.ax_avg = plt.subplots(figsize=(4, 3))
        with self.out_compl:
            #
            self.fig_compl, self.ax_compl = plt.subplots(figsize=(4, 3))

        options = ["displacement", "cumulative displacement"]
        for m in range(len(self.data.signalfile)):
            options.append(self.data.get_channel_name(m))

        # update signal 1
        self.signal1_selector = ipw.RadioButtons(
            options=options, value="displacement", description="Signal 1:"
        )

        def signal1_change(change):
            self.f1 = self.get_signal(change["new"], mode_selector.value)
            self.update_plots()

        self.signal1_selector.observe(signal1_change, names="value")

        self.signal2_selector = ipw.RadioButtons(
            options=options, value="displacement", description="Signal 2:"
        )

        def signal2_change(change):
            self.f2 = self.get_signal(change["new"], mode_selector.value)
            self.update_plots()

        self.signal2_selector.observe(signal2_change, names="value")

        mode_selector = ipw.RadioButtons(
            options=["Mean", "Variance"], value="Mean", description="Mode:"
        )

        def mode_change(change):
            self.f1 = self.get_signal(self.signal1_selector.value, change["new"])
            self.f2 = self.get_signal(self.signal2_selector.value, change["new"])
            self.update_plots()

        mode_selector.observe(mode_change, names="value")

        self.export_button = ipw.Button(
            description="Export as CSV", disabled=False, button_style=""
        )

        def export_as_csv(change):
            c = correlate_arrays(self.f1, self.f2, "Pearson")
            np.savetxt(
                os.path.join(self.param.resultdir, "Correlation.csv"),
                c,
                delimiter=",",
            )

        self.export_button.on_click(export_as_csv)

        self.window_slider = ipw.IntRangeSlider(
            description="Window range",
            min=0,
            max=self.res.I[0] - 1,
            value=[0, self.res.I[0] - 1],
            style={"description_width": "initial"},
            layout=ipw.Layout(width="100%"),
            continuous_update=False,
        )

        def window_change(change):
            self.update_plots()

        self.window_slider.observe(window_change, names="value")

        self.f1 = self.get_signal(self.signal1_selector.value, mode_selector.value)
        self.f2 = self.get_signal(self.signal2_selector.value, mode_selector.value)
        self.update_plots()

        self.interface = ipw.VBox(
            [
                ipw.HBox(
                    [
                        self.signal1_selector,
                        self.signal2_selector,
                        ipw.VBox([mode_selector, self.export_button]),
                    ]
                ),
                self.window_slider,
                ipw.HBox([self.out, ipw.VBox([self.out_avg, self.out_compl])]),
            ]
        )

    def update_plots(self):
        with self.out:
            self.fig = self.show_correlation(self.fig)
        with self.out_avg:
            self.fig_avg, _ = self.show_correlation_average(
                self.f1,
                self.f2,
                self.signal1_selector.value,
                self.signal2_selector.value,
                self.window_slider.value,
                self.fig_avg,
            )
        with self.out_compl:
            self.fig_compl, _ = self.show_correlation_compl(
                self.f1,
                self.f2,
                self.signal1_selector.value,
                self.signal2_selector.value,
                self.window_slider.value,
                self.fig_compl,
            )

    def get_signal(self, name, mode):
        if name == "displacement":
            return self.res.displacement
        if name == "cumulative displacement":
            return np.cumsum(self.res.displacement, axis=1)
        for m in range(len(self.data.signalfile)):
            if name == self.data.get_channel_name(m):
                if mode == "Mean":
                    return self.res.mean[m, 0]
                elif mode == "Variance":
                    return self.res.var[m, 0]

    def show_correlation(self, fig=None):

        c = correlate_arrays(self.f1, self.f2, "Pearson")
        fig, ax = show_correlation_core(
            c,
            self.f1,
            self.f2,
            self.signal1_selector.value,
            self.signal2_selector.value,
            "Pearson",
            fig,
        )
        ax.set_title(ax.get_title(), fontsize=10)
        return fig

    def show_correlation_average(
        self, f1, f2, f1_name, f2_name, range, fig=None, ax=None
    ):

        if fig is None:
            fig, ax = plt.subplots()
        else:
            plt.figure(fig.number)
            ax = fig.axes[0]
            ax.clear()
        # plt.clf()
        ax.set_title(
            "Average correlation between "
            + f1_name
            + " and \n"
            + f2_name
            + " - Windows "
            + str(range[0])
            + " to "
            + str(range[1]),
            fontsize=10,
        )
        t = get_range(f1.shape[1], f2.shape[1])
        c = correlate_arrays(
            f1[range[0] : range[1] + 1], f2[range[0] : range[1] + 1], "Pearson"
        )
        ax.plot(t, np.mean(c, axis=0))
        ax.plot(
            t,
            self.compute_significance_level(
                f1.shape[0], f1.shape[1], f2.shape[1], f1_name == f2_name
            ),
            "k--",
        )
        A, B = self.compute_confidence_interval(c)
        # plt.plot(t, A, 'g')
        # plt.plot(t, B, 'g')
        ax.fill_between(t, A, B, facecolor="orange", color="red", alpha=0.2)
        # plt.grid()
        ax.set_xlabel("Time lag [frames]")
        ax.set_ylabel("Correlation")
        plt.tight_layout()

        return fig, ax

    def show_correlation_compl(self, f1, f2, f1_name, f2_name, r, fig=None, ax=None):

        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
            plt.figure(fig.number)
            ax.clear()
            # if len(fig.axes) == 2:
            #    fig.axes[1].clear()

        i = np.concatenate(
            (
                np.array(range(0, r[0]), dtype=np.int),
                np.array(range(r[1] + 1, self.res.I[0]), dtype=np.int),
            )
        )
        if len(i) > 0:
            # plt.clf()
            ax.set_title(
                "Average correlation between "
                + f1_name
                + " and \n"
                + f2_name
                + " - Windows 0 to "
                + str(r[0] - 1)
                + " and "
                + str(r[1] + 1)
                + " to "
                + str(self.res.I[0]),
                fontsize=10,
            )
            t = get_range(f1.shape[1], f2.shape[1])
            c = correlate_arrays(f1[i], f2[i], "Pearson")
            ax.plot(t, np.mean(c, axis=0))
            ax.plot(
                t,
                self.compute_significance_level(
                    len(i), f1.shape[1], f2.shape[1], f1_name == f2_name
                ),
                "k--",
            )
            A, B = self.compute_confidence_interval(c)
            ax.fill_between(t, A, B, facecolor="orange", color="red", alpha=0.2)
            # plt.grid()
            ax.set_xlabel("Time lag [frames]")
            ax.set_ylabel("Correlation")
            plt.tight_layout()

        return fig, ax

    def compute_significance_level(self, M, K1, K2, autocorrelation=False):
        np.random.seed(15943)
        N = 1000
        s = np.zeros((N, K1 + K2 - 1))
        for n in range(N):
            x = np.random.randn(M, K1)
            if autocorrelation:
                y = x
            else:
                y = np.random.randn(M, K2)
            c = correlate_arrays(x, y, "Pearson")
            s[n] = np.mean(c, axis=0)
        return np.percentile(s, 95, axis=0)

    def compute_confidence_interval(self, rho):
        np.random.seed(22304)
        alpha = 0.025
        M = rho.shape[0]
        N = 1000
        i = np.random.randint(0, M, (N, M))
        idx = np.abs(rho) > 0.99999
        rho[idx] = np.sign(rho[idx]) * 0.99999
        R = np.arctanh(rho)  # Expecting problems if correlation equals +1 or -1
        T = np.mean(R, axis=0)
        Tbs = np.zeros((N, rho.shape[1]))
        for n in range(N):
            Tbs[n] = np.mean(rho[i[n, :]], axis=0)
        beta = np.mean(Tbs, axis=0) - T
        v = np.var(
            Tbs, axis=0, ddof=1
        )  # Normalization by N-1, as in Matlab's bootci function
        A = np.tanh(T - beta - v ** 0.5 * norm.ppf(1 - alpha))
        B = np.tanh(T - beta - v ** 0.5 * norm.ppf(alpha))
        return A, B
