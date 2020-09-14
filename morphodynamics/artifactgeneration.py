import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as ipw
from IPython.display import display
from matplotlib.backends.backend_pdf import PdfPages
from tifffile import TiffWriter, imsave
from scipy.interpolate import splev
from scipy.stats import norm
from PIL import Image
from .correlation import show_correlation_core, correlate_arrays, get_range
from .displacementestimation import show_edge_line, show_edge_image, compute_curvature, compute_length, compute_area, \
    splevper, show_edge_scatter_init, show_edge_scatter_update, show_edge_scatter
from .settings import Struct


def show_circularity(param, data, res, export=False, size=(16, 9)):
    if export:
        pp = PdfPages(os.path.join(param.resultdir, 'Circularity.pdf'))
    else:
        pp = None

    length = np.zeros((data.K,))
    area = np.zeros((data.K,))
    for k in range(data.K):
        length[k] = compute_length(res.spline[k])  # Length of the contour
        area[k] = compute_area(res.spline[k])  # Area delimited by the contour

    plt.figure(figsize=size)
    plt.gca().set_title('Length')
    plt.plot(length)
    plt.tight_layout()
    if export:
        pp.savefig()

    plt.figure(figsize=size)
    plt.gca().set_title('Area')
    plt.plot(area)
    plt.tight_layout()
    if export:
        pp.savefig()

    plt.figure(figsize=size)
    plt.gca().set_title('Circularity = Length^2 / Area / 4 / pi')
    plt.plot(length ** 2 / area / 4 / math.pi)
    plt.tight_layout()
    if export:
        pp.savefig()

    if export:
        pp.close()


def show_edge_overview(param, data, res, export=False, size=(12,9)):
    plt.figure(figsize=size)
    plt.gca().set_title('Edge overview')
    plt.imshow(data.load_frame_morpho(0), cmap='gray')
    show_edge_line(res.spline)
    plt.tight_layout()
    if export:
        plt.savefig(os.path.join(param.resultdir, 'Edge overview.pdf'))


def show_edge_vectorial_aux(data, res, k, curvature=False):
    plt.clf()
    plt.gca().set_title('Frame ' + str(k) + ' to frame ' + str(k + 1))
    plt.imshow(data.load_frame_morpho(k), cmap='gray')
    # showWindows(w, find_boundaries(labelWindows(w0)))  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
    if curvature:
        f = compute_curvature(res.spline[k], np.linspace(0, 1, 10001))
    else:
        f = res.displacement[:, k]
    show_edge_scatter(res.spline[k], res.spline[k + 1], res.param0[k], res.param[k], f)  # Show edge structures (spline curves, displacement vectors/curvature)
    plt.tight_layout()


def show_edge_vectorial(param, data, res, curvature=False, size=(12, 9)):
    if curvature:
        name = 'Edge animation with curvature'
    else:
        name = 'Edge animation with displacement'

    plt.figure(figsize=size)
    pp = PdfPages(os.path.join(param.resultdir, name + '.pdf'))

    plt.text(0.5, 0.5, 'This page intentionally left blank.')
    pp.savefig()

    # dmax = np.max(np.abs(res.displacement))
    for k in range(data.K - 1):
        print(k)
        show_edge_vectorial_aux(data, res, k, curvature)
        pp.savefig()
    pp.close()


class EdgeVectorialSlow():
    def __init__(self, param, data, res):
        self.param = param
        self.data = data
        self.res = res
        self.mode = 'displacement'

    def create_interface(self):
        out = ipw.Output()

        mode_selector = ipw.RadioButtons(
            options=['displacement', 'curvature'],
            value='displacement',
            description='Mode:'
        )

        def mode_change(change):
            with out:
                self.set_mode(change['new'])
                plt.figure(self.fig.number)
                show_edge_vectorial_aux(self.data, self.res, time_slider.get_state()['value'], curvature=self.curvature)

        mode_selector.observe(mode_change, names='value')

        time_slider = ipw.IntSlider(description='Time', min=0, max=self.data.K - 2, continuous_update=False, layout=ipw.Layout(width='100%'))

        def time_change(change):
            with out:
                plt.figure(self.fig.number)
                show_edge_vectorial_aux(self.data, self.res, change['new'], curvature=self.curvature)

        time_slider.observe(time_change, names='value')

        # display(mode_selector)
        display(time_slider)

        self.set_mode(mode_selector.get_state('value'))

        self.fig = plt.figure(figsize=(8, 6))
        show_edge_vectorial_aux(self.data, self.res, 0, curvature=False)
        display(out)

    def set_mode(self, mode):
        self.curvature = mode == 'curvature'


class EdgeVectorial():
    def __init__(self, param, data, res):
        self.param = param
        self.data = data
        self.res = res
        self.mode = 'displacement'

    def create_interface(self):
        out = ipw.Output()

        mode_selector = ipw.RadioButtons(
            options=['displacement', 'curvature'],
            value='displacement',
            description='Mode:'
        )

        def mode_change(change):
            with out:
                self.set_mode(change['new'])
                plt.figure(self.fig.number)
                self.show_edge_vectorial_aux_update(self.data, self.res, time_slider.get_state()['value'], curvature=self.curvature)

        mode_selector.observe(mode_change, names='value')

        time_slider = ipw.IntSlider(description='Time', min=0, max=self.data.K - 1, continuous_update=False, layout=ipw.Layout(width='100%'))

        def time_change(change):
            with out:
                plt.figure(self.fig.number)
                self.show_edge_vectorial_aux_update(self.data, self.res, change['new'], curvature=self.curvature)

        time_slider.observe(time_change, names='value')

        # display(mode_selector)
        display(time_slider)

        self.set_mode(mode_selector.get_state('value'))

        self.fig = plt.figure(figsize=(8, 6))
        self.show_edge_vectorial_aux_init(self.data, self.res, 0, curvature=False)
        display(out)

    def show_edge_vectorial(self, param, data, res, curvature=False, size=(12, 9)):
        if curvature:
            name = 'Edge animation with curvature'
        else:
            name = 'Edge animation with displacement'

        plt.figure(figsize=size)
        pp = PdfPages(os.path.join(param.resultdir, name + '.pdf'))

        plt.text(0.5, 0.5, 'This page intentionally left blank.')
        pp.savefig()

        # dmax = np.max(np.abs(res.displacement))
        self.show_edge_vectorial_aux_init(data, res, 0, curvature)
        for k in range(1, data.K - 1):
            print(k)
            self.show_edge_vectorial_aux_update(data, res, k, curvature)
            pp.savefig()
        pp.close()

    def show_edge_vectorial_aux_init(self, data, res, k, curvature=False):
        plt.clf()
        self.p = Struct()
        self.p.t = plt.gca().set_title('Frame ' + str(k) + ' to frame ' + str(k + 1))
        self.p.i = plt.imshow(data.load_frame_morpho(k), cmap='gray')
        # showWindows(w, find_boundaries(labelWindows(w0)))  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
        if curvature:
            f = compute_curvature(res.spline[k], np.linspace(0, 1, 10001))
        else:
            f = res.displacement[:, k]
        self.p = show_edge_scatter_init(self.p, res.spline[k], res.spline[k + 1], res.param0[k], res.param[k], f)  # Show edge structures (spline curves, displacement vectors/curvature)
        plt.tight_layout()

    def show_edge_vectorial_aux_update(self, data, res, k, curvature=False):
        self.p.t.set_text('Frame ' + str(k) + ' to frame ' + str(k + 1))
        self.p.i.set_data(data.load_frame_morpho(k))
        # showWindows(w, find_boundaries(labelWindows(w0)))  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
        if curvature:
            f = compute_curvature(res.spline[k], np.linspace(0, 1, 10001))
        else:
            f = res.displacement[:, k]
        self.p = show_edge_scatter_update(self.p, res.spline[k], res.spline[k + 1], res.param0[k], res.param[k], f)  # Show edge structures (spline curves, displacement vectors/curvature)
        plt.tight_layout()

    def set_mode(self, mode):
        self.curvature = mode == 'curvature'

    # def show_edge_scatter(s1, s2, t1, t2, d, dmax=None):
    #     """ Draw the cell-edge contour and the displacement vectors.
    #     The contour is drawn using a scatter plot to color-code the displacements. """
    #
    #     # Evaluate splines at window locations and on fine-resolution grid
    #     c1 = splevper(t1, s1)
    #     c2 = splevper(t2, s2)
    #     c1p = splev(np.linspace(0, 1, 10001), s1)
    #     c2p = splev(np.linspace(0, 1, 10001), s2)
    #
    #     # Interpolate displacements
    #     # d = 0.5 + 0.5 * d / np.max(np.abs(d))
    #     if len(d) < 10001:
    #         d = np.interp(np.linspace(0, 1, 10001), t1, d, period=1)
    #     if dmax is None:
    #         dmax = np.max(np.abs(d))
    #         if dmax == 0:
    #             dmax = 1
    #
    #     # Plot results
    #     # matplotlib.use('PDF')
    #     lw = 1
    #     s = 1  # Scaling factor for the vectors
    #
    #     plt.plot(c1p[0], c1p[1], 'b', zorder=50, lw=lw)
    #     plt.plot(c2p[0], c2p[1], 'r', zorder=100, lw=lw)
    #     # plt.scatter(c1p[0], c1p[1], c=d, cmap='bwr', vmin=-dmax, vmax=dmax, zorder=50, s1=lw)
    #     # # plt.colorbar(label='Displacement [pixels]')
    #     for j in range(len(t2)):
    #         plt.arrow(c1[0][j], c1[1][j], s * (c2[0][j] - c1[0][j]), s * (c2[1][j] - c1[1][j]), color='y', zorder=200, lw=lw)
    #     # plt.arrow(c1[0][j], c1[1][j], s1 * u[0][j], s1 * u[1][j], color='y', zorder=200, lw=lw) # Show normal to curve
    #     plt.arrow(c1[0][0], c1[1][0], s * (c2[0][0] - c1[0][0]), s * (c2[1][0] - c1[1][0]), color='c', zorder=400, lw=lw)
    #
    # def show_edge_vectorial_aux(data, res, k, curvature=False):
    #     plt.clf()
    #     plt.gca().set_title('Frame ' + str(k) + ' to frame ' + str(k + 1))
    #     plt.imshow(data.load_frame_morpho(k), cmap='gray')
    #     # showWindows(w, find_boundaries(labelWindows(w0)))  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
    #     if curvature:
    #         f = compute_curvature(res.spline[k])
    #     else:
    #         f = res.displacement[:, k]
    #     show_edge_scatter(res.spline[k], res.spline[k + 1], res.param0[k], res.param[k], f)  # Show edge structures (spline curves, displacement vectors/curvature)
    #     plt.tight_layout()
    #
    #
    # def show_edge_vectorial(param, data, res, curvature=False, size=(12, 9)):
    #     if curvature:
    #         name = 'Edge animation with curvature'
    #     else:
    #         name = 'Edge animation with displacement'
    #
    #     plt.figure(figsize=size)
    #     pp = PdfPages(param.resultdir + name + '.pdf')
    #
    #     plt.text(0.5, 0.5, 'This page intentionally left blank.')
    #     pp.savefig()
    #
    #     # dmax = np.max(np.abs(res.displacement))
    #     for k in range(data.K - 1):
    #         print(k)
    #         show_edge_vectorial_aux(data, res, k, curvature)
    #         pp.savefig()
    #     pp.close()


def show_edge_rasterized_aux(data, res, d, dmax, k, mode, display=True):
    if mode == 'curvature':
        x = show_edge_image(data.shape, res.spline[k], np.linspace(0, 1, 10000, endpoint=False), d[:, k], 3, dmax)
    else:
        x = show_edge_image(data.shape, res.spline[k], res.param0[k], d[:, k], 3, dmax)

    if display:
        plt.clf()
        plt.gca().set_title('Frame ' + str(k))
        plt.imshow(x)
        plt.tight_layout()

    return x


def show_edge_rasterized(param, data, res, mode=None):
    if mode=='cumulative':
        d = np.cumsum(res.displacement, axis=1)
        name = 'Edge animation (cumulative)'
    elif mode == 'simple':
        d = np.ones(res.displacement.shape)
        name = 'Edge animation (simple)'
    elif mode == 'curvature':
        t = np.linspace(0, 1, 10000, endpoint=False)
        d = np.zeros((10000, data.K))
        for k in range(data.K):
            d[:, k] = compute_curvature(res.spline[k], t)
        name = 'Edge animation (curvature)'
    else:
        d = res.displacement
        name = 'Edge animation'

    if param.edgeNormalization == 'global':
        dmax = np.max(np.abs(d))
    else:
        dmax = None

    y = np.zeros((data.K,) + data.shape)
    for k in range(data.K):
        y[k] = data.load_frame_morpho(k)
    y = 255 * y / np.max(y)

    tw = TiffWriter(os.path.join(param.resultdir, name + '.tif'))
    for k in range(data.K - 1):
        x = show_edge_rasterized_aux(data, res, d, dmax, k, mode, display=False)
        y0 = np.stack((y[k], y[k], y[k]), axis=-1)
        x[x==0] = y0[x==0]
        tw.save(x, compress=6)
    tw.close()


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
            options=['border', 'displacement', 'cumulative displacement', 'curvature'],
            value='displacement',
            description='Mode:'
        )

        def mode_change(change):
            with out:
                self.set_mode(change['new'])
                self.set_normalization(normalization_selector.value)
                self.plot(time_slider.value)

        mode_selector.observe(mode_change, names='value')

        normalization_selector = ipw.RadioButtons(
            options=['global', 'frame-by-frame'],
            value='frame-by-frame',
            layout={'width': 'max-content'}, # If the items' names are long
            description='Normalization:'
            # disabled=False
        )

        def normalization_change(change):
            with out:
                self.set_normalization(change['new'])
                self.plot(time_slider.value)

        normalization_selector.observe(normalization_change, names='value')

        time_slider = ipw.IntSlider(description='Time', min=0, max=self.data.K - 1, continuous_update=False, layout=ipw.Layout(width='100%'))

        def time_change(change):
            with out:
                self.plot(change['new'])

        time_slider.observe(time_change, names='value')

        display(mode_selector)
        display(normalization_selector)
        display(time_slider)

        self.set_mode(mode_selector.value)
        self.set_normalization(normalization_selector.value)

        self.fig = plt.figure(figsize=(8, 6))
        self.plot(0)
        display(out)

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'border':
            self.d = np.ones(self.res.displacement.shape)
            self.name = 'Edge animation (border)'
        elif mode == 'displacement':
            self.d = self.res.displacement
            self.name = 'Edge animation (displacement)'
        elif mode == 'cumulative displacement':
            self.d = np.cumsum(self.res.displacement, axis=1)
            self.name = 'Edge animation (cumulative displacement)'
        elif mode == 'curvature':
            t = np.linspace(0, 1, 10000, endpoint=False)
            self.d = np.zeros((10000, self.data.K))
            for k in range(self.data.K):
                self.d[:, k] = compute_curvature(self.res.spline[k], t)
            self.name = 'Edge animation (curvature)'

    def set_normalization(self, normalization):
        if normalization == 'global':
            self.dmax = np.max(np.abs(self.d))
        else:
            self.dmax = None

    def get_image(self, k):
        x = show_edge_rasterized_aux(self.data, self.res, self.d, self.dmax, k, self.mode, display=False)
        y0 = np.stack((self.y[k], self.y[k], self.y[k]), axis=-1)
        x[x == 0] = y0[x == 0]
        return x

    def save(self):
        tw = TiffWriter(self.param.resultdir + self.name + '.tif')
        for k in range(self.data.K - 1):
            tw.save(self.get_image(k), compress=6)
        tw.close()

    def plot(self, k):
        plt.figure(self.fig.number)
        plt.clf()
        plt.gca().set_title('Frame ' + str(k))
        plt.imshow(self.get_image(k))
        plt.tight_layout()


def show_curvature(param, data, res, size=(16, 9), cmax=None, export=True):
    curvature = np.zeros((10000, data.K))
    for k in range(data.K):
        curvature[:, k] = compute_curvature(res.spline[k], np.linspace(0, 1, 10000, endpoint=False))

    plt.figure(figsize=size)
    plt.gca().set_title('Curvature')
    if export:
        pp = PdfPages(os.path.join(param.resultdir, 'Curvature.pdf'))
    if cmax is None:
        cmax = np.max(np.abs(curvature))
    plt.imshow(curvature, cmap='seismic', vmin=-cmax, vmax=cmax)
    plt.colorbar(label='Curvature')
    plt.axis('auto')
    plt.xlabel('Frame index')
    plt.ylabel('Position on contour')
    # imsave(param.resultdir + 'Curvature.tif', curvature.astype(np.float32), compress=6)
    if export:
        pp.savefig()
        pp.close()


class Curvature:
    def __init__(self, param, data, res):
        self.param = param
        self.data = data
        self.res = res

        self.curvature = np.zeros((10000, self.data.K))
        for k in range(self.data.K):
            self.curvature[:, k] = compute_curvature(self.res.spline[k], np.linspace(0, 1, 10000, endpoint=False))

    def create_interface(self):
        out = ipw.Output()

        cmax = np.max(np.abs(self.curvature))

        range_slider = ipw.FloatSlider(value = cmax, description='Range:', min=0, max=cmax, step=0.1, continuous_update=False, layout=ipw.Layout(width='100%'))

        def range_change(change):
            with out:
                plt.figure(self.fig.number)
                self.show_curvature(change['new'])

        range_slider.observe(range_change, names='value')

        display(range_slider)

        self.fig = plt.figure(figsize=(8, 6))
        self.show_curvature(range_slider.value)
        display(out)

    def show_curvature(self, cmax, export=True):
        plt.clf()
        plt.gca().set_title('Curvature')
        if export:
            pp = PdfPages(os.path.join(self.param.resultdir, 'Curvature.pdf'))
        plt.imshow(self.curvature, cmap='seismic', vmin=-cmax, vmax=cmax)
        plt.colorbar(label='Curvature')
        plt.axis('auto')
        plt.xlabel('Frame index')
        plt.ylabel('Position on contour')
        # imsave(param.resultdir + 'Curvature.tif', curvature.astype(np.float32), compress=6)
        if export:
            pp.savefig()
            pp.close()


def show_displacement(param, res, size=(16, 9), export=True):
    
    if export:
        pp = PdfPages(os.path.join(param.resultdir, 'Displacement.pdf'))

    plt.figure(figsize=size)
    plt.gca().set_title('Displacement')
    plt.imshow(res.displacement, cmap='seismic')
    plt.axis('auto')
    plt.xlabel('Frame index')
    plt.ylabel('Window index')
    plt.colorbar(label='Displacement [pixels]')
    if hasattr(param, 'scaling_disp'):
        cmax = param.scaling_disp
    else:
        cmax = np.max(np.abs(res.displacement))
    plt.clim(-cmax, cmax)
    # plt.xticks(range(0, velocity.shape[1], 5))
    if export:
        pp.savefig()
    # imsave('Displacement.tif', res.displacement.astype(np.float32))

    dcum = np.cumsum(res.displacement, axis=1)

    plt.figure(figsize=size)
    plt.gca().set_title('Cumulative displacement')
    plt.imshow(dcum, cmap='seismic')
    plt.axis('auto')
    plt.xlabel('Frame index')
    plt.ylabel('Window index')
    plt.colorbar(label='Displacement [pixels]')
    cmax = np.max(np.abs(dcum))
    plt.clim(-cmax, cmax)
    # plt.xticks(range(0, velocity.shape[1], 5))
    if export:
        pp.savefig()

        pp.close()


def show_signals_aux(param, data, res, m, j, mode):
    if mode == 'Mean':
        f = res.mean[m, j, 0:res.I[j], :]
    elif mode == 'Variance':
        f = res.var[m, j, 0:res.I[j], :]

    plt.clf()
    plt.gca().set_title('Signal: ' + data.get_channel_name(m) + ' - Layer: ' + str(j))
    # if hasattr(param, 'scaling_mean') & (j == 0):
    #     plt.imshow(res.mean[m, j, 0:res.I[j], :], cmap='jet', vmin=param.scaling_mean[m][0], vmax=param.scaling_mean[m][1])
    # else:
    plt.imshow(f, cmap='jet')
    plt.colorbar(label=mode)
    plt.axis('auto')
    plt.xlabel('Frame index')
    plt.ylabel('Window index')
    # plt.xticks(range(0, mean.shape[1], 5))


def show_signals(param, data, res, mode, size=(16, 9), export=True):
    f = plt.figure(figsize=size)

    if export:
        pp = PdfPages(os.path.join(param.resultdir, 'Signal ' + mode + '.pdf'))
    for m in range(len(data.signalfile)):
        for j in range(res.mean.shape[1]):
            plt.figure(f.number)
            show_signals_aux(param, data, res, m, j, mode)
            # if j == 0:
            #     imsave(param.resultdir + 'Signal ' + mode + ' ' + data.get_channel_name(m) + '.tif', res.mean[m, j, 0:res.I[j], :].astype(np.float32), compress=6)
            if export:
                pp.savefig()
    if export:
        pp.close()

    # pp = PdfPages(param.resultdir + 'Signal variance.pdf')
    # # tw = TiffWriter(param.resultdir + 'Variance.tif')
    # for m in range(len(data.signalfile)):
    #     for j in range(res.mean.shape[1]):
    #         plt.figure(f.number)
    #         plt.clf()
    #         plt.gca().set_title('Signal: ' + data.get_channel_name(m) + ' - Layer: ' + str(j))
    #         plt.imshow(res.var[m, j, 0:res.I[j], :], cmap='jet')
    #         plt.colorbar(label='Variance')
    #         plt.axis('auto')
    #         plt.xlabel('Frame index')
    #         plt.ylabel('Window index')
    #         # plt.xticks(range(0, mean.shape[1], 5))
    #         if j == 0:
    #             imsave(param.resultdir + 'Signal variance ' + data.get_channel_name(m) + '.tif', res.var[m, j, 0:res.I[j], :].astype(np.float32), compress=6)
    #         # if j == 0:
    #         #     tw.save(res.var[m, j, 0:param.I//2**j, :], compress=6)
    #         pp.savefig()
    # pp.close()
    # # tw.close()


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
            description='Signal:'
        )

        def channel_change(change):
            plt.figure(self.fig.number)
            show_signals_aux(self.param, self.data, self.res, self.get_signal(change['new']), layer_text.value, mode_selector.value)

        signal_selector.observe(channel_change, names='value')

        layer_text = ipw.BoundedIntText(value=0, min=0, max=self.res.J-1, description = 'Layer:')

        def layer_change(change):
            with out:
                plt.figure(self.fig.number)
                show_signals_aux(self.param, self.data, self.res, self.get_signal(signal_selector.value), layer_text.value, mode_selector.value)

        layer_text.observe(layer_change, names='value')

        mode_selector = ipw.RadioButtons(
            options=['Mean', 'Variance'],
            value='Mean',
            description='Mode:'
        )

        def mode_change(change):
            with out:
                plt.figure(self.fig.number)
                show_signals_aux(self.param, self.data, self.res, self.get_signal(signal_selector.value), layer_text.value, mode_selector.value)

        mode_selector.observe(mode_change, names='value')

        display(signal_selector)
        display(layer_text)
        display(mode_selector)

        self.fig = plt.figure(figsize=(8, 6))
        show_signals_aux(self.param, self.data, self.res, self.get_signal(signal_selector.value), layer_text.value, mode_selector.value)
        display(out)


def show_fourier_descriptors_aux(res, k):
    N = 1000
    c = splev(np.linspace(0, 1, N, endpoint=False), res.spline[k])
    c = c[0] + 1j * c[1]
    chat = np.fft.fftshift(np.fft.fft(c))
    plt.clf()
    plt.gca().set_title('Fourier descriptors at frame ' + str(k))
    plt.plot(range(-N // 2, N // 2), np.log(np.abs(chat)))
    plt.ylim(-5, 15)
    plt.xlabel('Frequency')
    plt.ylabel('log(abs(Fourier coefficients))')


def show_fourier_descriptors(param, data, res, size=(16, 9)):
    plt.figure(figsize=size)
    pp = PdfPages(os.path.join(param.resultdir, 'Fourier descriptors.pdf'))
    for k in range(data.K):
        show_fourier_descriptors_aux(res, k)
        pp.savefig()
    pp.close()

def show_correlation(param, data, res, size=(16, 9)):
    dcum = np.cumsum(res.displacement, axis=1)

    plt.figure(figsize=size)

    pp = PdfPages(os.path.join(param.resultdir, "Correlation mean.pdf"))
    c = correlate_arrays(res.displacement, res.displacement, 'Pearson')
    show_correlation_core(c, res.displacement, res.displacement, 'displacement', 'displacement', 'Pearson')
    pp.savefig()
    # show_average_correlation(fh, c, res.displacement, res.displacement)
    # pp.savefig()
    for m in range(len(data.signalfile)):
        c = correlate_arrays(res.displacement, res.mean[m, 0], 'Pearson')
        show_correlation_core(c, res.displacement, res.mean[m, 0], 'displacement', data.get_channel_name(m), 'Pearson')
        pp.savefig()
        # show_average_correlation(fh, c, res.displacement, res.mean[m, 0])
        # pp.savefig()
        c = correlate_arrays(dcum, res.mean[m, 0], 'Pearson')
        show_correlation_core(c, dcum, res.mean[m, 0], 'cumulative displacement', data.get_channel_name(m), 'Pearson')
        pp.savefig()
        # show_average_correlation(fh, c, dcum, res.mean[m, 0])
        # pp.savefig()
    for m in range(len(data.signalfile)):
        for mprm in range(m + 1, len(data.signalfile)):
            c = correlate_arrays(res.mean[m, 0], res.mean[mprm, 0], 'Pearson')
            show_correlation_core(c, res.mean[m, 0], res.mean[mprm, 0], data.get_channel_name(m), data.get_channel_name(mprm), 'Pearson')
            pp.savefig()
            # show_average_correlation(fh, c, res.mean[0, 0], res.mean[-1, 0])
            # pp.savefig()
    pp.close()

    pp = PdfPages(os.path.join(param.resultdir, "Correlation variance.pdf"))
    # c = correlate_arrays(res.displacement, res.displacement, 'Pearson')
    # show_correlation_core(c, res.displacement, res.displacement, 'displacement', 'displacement', 'Pearson')
    # pp.savefig()
    # show_average_correlation(fh, c, res.displacement, res.displacement)
    # pp.savefig()
    for m in range(len(data.signalfile)):
        c = correlate_arrays(res.displacement, res.var[m, 0], 'Pearson')
        show_correlation_core(c, res.displacement, res.var[m, 0], 'displacement', data.get_channel_name(m), 'Pearson')
        pp.savefig()
        # show_average_correlation(fh, c, res.displacement, res.var[m, 0])
        # pp.savefig()
        c = correlate_arrays(dcum, res.var[m, 0], 'Pearson')
        show_correlation_core(c, dcum, res.var[m, 0], 'cumulative displacement', data.get_channel_name(m), 'Pearson')
        pp.savefig()
        # show_average_correlation(fh, c, dcum, res.var[m, 0])
        # pp.savefig()
    for m in range(len(data.signalfile)):
        for mprm in range(m + 1, len(data.signalfile)):
            c = correlate_arrays(res.var[m, 0], res.var[mprm, 0], 'Pearson')
            show_correlation_core(c, res.var[m, 0], res.var[mprm, 0], data.get_channel_name(m), data.get_channel_name(mprm), 'Pearson')
            pp.savefig()
            # show_average_correlation(fh, c, res.var[0, 0], res.var[-1, 0])
            # pp.savefig()
    pp.close()


def show_correlation_average(param, data, res, size=(16, 9), export=True):
    dcum = np.cumsum(res.displacement, axis=1)

    plt.figure(figsize=size)

    if export:
        pp = PdfPages(os.path.join(param.resultdir, "Correlation average.pdf"))
    plt.clf()
    plt.gca().set_title('Average cross-correlation between displacement and signals at layer ' + str(0))
    color = 'rbgymc'
    n = 0
    for m in [0, len(data.signalfile) - 1]:
        t = get_range(res.displacement.shape[1], res.mean[m, 0].shape[1])
        c = correlate_arrays(res.displacement, res.mean[m, 0], 'Pearson')
        plt.plot(t, np.mean(c, axis=0), color[n], label=data.get_channel_name(m))
        n += 1
    plt.grid()
    plt.legend(loc="upper left")
    plt.xlabel('Time lag [frames]')
    plt.ylabel('Cross-correlation')
    if export:
        pp.savefig()

    # plt.clf()
    # plt.gca().set_title('Average cross-correlation between displacement and signals at layer ' + str(0) + ' - Windows 40 to 59')
    # color = 'rbgymc'
    # n = 0
    # for m in [0, len(data.signalfile) - 1]:
    #     t = get_range(res.displacement.shape[1], res.mean[m, 0].shape[1])
    #     c = correlate_arrays(res.displacement[40:60], res.mean[m, 0][40:60], 'Pearson')
    #     plt.plot(t, np.mean(c, axis=0), color[n], label=data.get_channel_name(m))
    #     n += 1
    # plt.grid()
    # plt.legend(loc="upper left")
    # plt.xlabel('Time lag [frames]')
    # plt.ylabel('Cross-correlation')
    # pp.savefig()

    plt.figure(figsize=size)
    plt.gca().set_title('Average cross-correlation between cumulative displacement and signals at layer ' + str(0))
    color = 'rbgymc'
    n = 0
    for m in [0, len(data.signalfile) - 1]:
        t = get_range(dcum.shape[1], res.mean[m, 0].shape[1])
        c = correlate_arrays(dcum, res.mean[m, 0], 'Pearson')
        plt.plot(t, np.mean(c, axis=0), color[n], label=data.get_channel_name(m))
        n += 1
    plt.grid()
    plt.legend(loc="upper left")
    plt.xlabel('Time lag [frames]')
    plt.ylabel('Cross-correlation')
    if export:
        pp.savefig()

    plt.figure(figsize=size)
    plt.gca().set_title('Average autocorrelation of displacement')
    t = get_range(res.displacement.shape[1], res.displacement.shape[1])
    c = correlate_arrays(res.displacement, res.displacement, 'Pearson')
    plt.plot(t, np.mean(c, axis=0), color[n])
    plt.grid()
    plt.xlabel('Time lag [frames]')
    plt.ylabel('Correlation')
    if export:
        pp.savefig()

    plt.figure(figsize=size)
    plt.gca().set_title('Average autocorrelation of cumulative displacement')
    t = get_range(dcum.shape[1], dcum.shape[1])
    c = correlate_arrays(dcum, dcum, 'Pearson')
    plt.plot(t, np.mean(c, axis=0), color[n])
    plt.grid()
    plt.xlabel('Time lag [frames]')
    plt.ylabel('Correlation')
    if export:
        pp.savefig()
        pp.close()


class Correlation():
    def __init__(self, param, data, res, mode=None):
        self.param = param
        self.data = data
        self.res = res

    def create_interface(self):
        out = ipw.Output()

        options = ['displacement', 'cumulative displacement']
        for m in range(len(self.data.signalfile)):
            options.append(self.data.get_channel_name(m))

        self.signal1_selector = ipw.RadioButtons(
            options=options,
            value='displacement',
            description='Signal 1:'
        )

        def signal1_change(change):
            with out:
                self.f1 = self.get_signal(change['new'], mode_selector.value)
                plt.figure(self.fig.number)
                self.show_correlation()
                plt.figure(self.fig_avg.number)
                self.show_correlation_average(self.f1, self.f2, self.signal1_selector.value, self.signal2_selector.value, self.window_slider.value)
                plt.figure(self.fig_compl.number)
                self.show_correlation_compl(self.f1, self.f2, self.signal1_selector.value, self.signal2_selector.value, self.window_slider.value)

        self.signal1_selector.observe(signal1_change, names='value')

        self.signal2_selector = ipw.RadioButtons(
            options=options,
            value='displacement',
            description='Signal 2:'
        )

        def signal2_change(change):
            with out:
                self.f2 = self.get_signal(change['new'], mode_selector.value)
                plt.figure(self.fig.number)
                self.show_correlation()
                plt.figure(self.fig_avg.number)
                self.show_correlation_average(self.f1, self.f2, self.signal1_selector.value, self.signal2_selector.value, self.window_slider.value)
                plt.figure(self.fig_compl.number)
                self.show_correlation_compl(self.f1, self.f2, self.signal1_selector.value, self.signal2_selector.value, self.window_slider.value)

        self.signal2_selector.observe(signal2_change, names='value')

        mode_selector = ipw.RadioButtons(
            options=['Mean', 'Variance'],
            value='Mean',
            description='Mode:'
        )

        def mode_change(change):
            with out:
                self.f1 = self.get_signal(self.signal1_selector.value, change['new'])
                self.f2 = self.get_signal(self.signal2_selector.value, change['new'])
                plt.figure(self.fig.number)
                self.show_correlation()
                plt.figure(self.fig_avg.number)
                self.show_correlation_average(self.f1, self.f2, self.signal1_selector.value, self.signal2_selector.value, self.window_slider.value)
                plt.figure(self.fig_compl.number)
                self.show_correlation_compl(self.f1, self.f2, self.signal1_selector.value, self.signal2_selector.value, self.window_slider.value)

        mode_selector.observe(mode_change, names='value')

        self.export_button = ipw.Button(
            description='Export as CSV',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            # tooltip='Click me',
            # icon='check'  # (FontAwesome names without the `fa-` prefix)
        )

        def export_as_csv(change):
            with out:
                c = correlate_arrays(self.f1, self.f2, 'Pearson')
                np.savetxt(os.path.join(self.param.resultdir, "Correlation.csv"), c, delimiter=",")

        self.export_button.on_click(export_as_csv)

        self.window_slider = ipw.IntRangeSlider(description='Window range', min=0, max=self.res.I[0]-1, value=[0, self.res.I[0]-1], style={"description_width": "initial"}, layout=ipw.Layout(width='100%'), continuous_update=False)
        # self.default_path_from_browser_button = ipw.Button(
        #     description="Update using browser",
        #     layout={"width": "200px"},
        #     style={"description_width": "initial"},
        # )

        def window_change(change):
            with out:
                plt.figure(self.fig_avg.number)
                self.show_correlation_average(self.f1, self.f2, self.signal1_selector.value, self.signal2_selector.value, self.window_slider.value)
                plt.figure(self.fig_compl.number)
                self.show_correlation_compl(self.f1, self.f2, self.signal1_selector.value, self.signal2_selector.value, self.window_slider.value)

        self.window_slider.observe(window_change, names='value')

        display(ipw.HBox([self.signal1_selector, self.signal2_selector, ipw.VBox([mode_selector, self.export_button])]))

        self.fig = plt.figure(figsize=(8, 6))
        self.f1 = self.get_signal(self.signal1_selector.value, mode_selector.value)
        self.f2 = self.get_signal(self.signal2_selector.value, mode_selector.value)
        self.show_correlation()

        display(self.window_slider)

        self.fig_avg = plt.figure(figsize=(8, 6))
        self.show_correlation_average(self.f1, self.f2, self.signal1_selector.value, self.signal2_selector.value, self.window_slider.value)
        self.fig_compl = plt.figure(figsize=(8, 6))
        self.show_correlation_compl(self.f1, self.f2, self.signal1_selector.value, self.signal2_selector.value, self.window_slider.value)

        display(out)

    def get_signal(self, name, mode):
        if name == 'displacement':
            return self.res.displacement
        if name == 'cumulative displacement':
            return np.cumsum(self.res.displacement, axis=1)
        for m in range(len(self.data.signalfile)):
            if name == self.data.get_channel_name(m):
                if mode =='Mean':
                    return self.res.mean[m, 0]
                elif mode == 'Variance':
                    return self.res.var[m, 0]

    def show_correlation(self):
        c = correlate_arrays(self.f1, self.f2, 'Pearson')
        show_correlation_core(c, self.f1, self.f2, self.signal1_selector.value, self.signal2_selector.value, 'Pearson')

    def show_correlation_average(self, f1, f2, f1_name, f2_name, range):
        plt.clf()
        plt.gca().set_title('Average correlation between ' + f1_name + ' and ' + f2_name + ' - Windows ' + str(range[0]) + ' to ' + str(range[1]))
        t = get_range(f1.shape[1], f2.shape[1])
        c = correlate_arrays(f1[range[0]:range[1]+1], f2[range[0]:range[1]+1], 'Pearson')
        plt.plot(t, np.mean(c, axis=0))
        plt.plot(t, self.compute_significance_level(f1.shape[0], f1.shape[1], f2.shape[1], f1_name==f2_name), 'k--')
        A, B = self.compute_confidence_interval(c)
        # plt.plot(t, A, 'g')
        # plt.plot(t, B, 'g')
        plt.fill_between(t, A, B,
                         facecolor="orange",  # The fill color
                         color='red',  # The outline color
                         alpha=0.2)  # Transparency of the fill
        plt.grid()
        plt.xlabel('Time lag [frames]')
        plt.ylabel('Correlation')

    def show_correlation_compl(self, f1, f2, f1_name, f2_name, r):
        i = np.concatenate((np.array(range(0, r[0]), dtype=np.int), np.array(range(r[1]+1, self.res.I[0]), dtype=np.int)))
        if len(i) > 0:
            plt.clf()
            plt.gca().set_title('Average correlation between ' + f1_name + ' and ' + f2_name + ' - Windows 0 to ' + str(r[0] - 1) + ' and ' + str(r[1] + 1) + ' to ' + str(self.res.I[0]))
            t = get_range(f1.shape[1], f2.shape[1])
            c = correlate_arrays(f1[i], f2[i], 'Pearson')
            plt.plot(t, np.mean(c, axis=0))
            plt.plot(t, self.compute_significance_level(len(i), f1.shape[1], f2.shape[1], f1_name == f2_name), 'k--')
            A, B = self.compute_confidence_interval(c)
            plt.fill_between(t, A, B,
                             facecolor="orange",  # The fill color
                             color='red',  # The outline color
                             alpha=0.2)  # Transparency of the fill
            plt.grid()
            plt.xlabel('Time lag [frames]')
            plt.ylabel('Correlation')

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
            c = correlate_arrays(x, y, 'Pearson')
            s[n] = np.mean(c, axis=0)
        return np.percentile(s, 95, axis=0)

    # def compute_confidence_interval(self, f1, f2, r):
    #     np.random.seed(22304)
    #     rho = correlate_arrays(f1[r[0]:r[1] + 1], f2[r[0]:r[1] + 1], 'Pearson')
    #     M = rho.shape[0]
    #     N = 1000
    #     i = np.random.randint(0, M, (N, M))
    #     rho_mean = np.zeros((N, rho.shape[1]))
    #     for n in range(N):
    #         rho_mean[n] = np.mean(rho[i[n, :]], axis=0)
    #     A = np.percentile(rho_mean, 2.5, axis=0)
    #     B = np.percentile(rho_mean, 97.5, axis=0)
    #     return A, B


    # def compute_confidence_interval(self, rho):
    #     np.random.seed(22304)
    #     M = rho.shape[0]
    #     N = 1000
    #     i = np.random.randint(0, M, (N, M))
    #     rho_mean = np.zeros((N, rho.shape[1]))
    #     for n in range(N):
    #         rho_mean[n] = np.mean(rho[i[n, :]], axis=0)
    #     A = np.percentile(rho_mean, 2.5, axis=0)
    #     B = np.percentile(rho_mean, 97.5, axis=0)
    #     return A, B


    def compute_confidence_interval(self, rho):
        np.random.seed(22304)
        alpha = 0.025
        M = rho.shape[0]
        N = 1000
        i = np.random.randint(0, M, (N, M))
        idx = np.abs(rho) > 0.99999
        rho[idx] = np.sign(rho[idx]) * 0.99999;
        R = np.arctanh(rho)  # Expecting problems if correlation equals +1 or -1
        T = np.mean(R, axis=0)
        Tbs = np.zeros((N, rho.shape[1]))
        for n in range(N):
            Tbs[n] = np.mean(rho[i[n, :]], axis=0)
        beta = np.mean(Tbs, axis=0) - T
        v = np.var(Tbs, axis=0, ddof=1)  # Normalization by N-1, as in Matlab's bootci function
        A = np.tanh(T-beta-v**0.5*norm.ppf(1-alpha))
        B = np.tanh(T-beta-v**0.5*norm.ppf(alpha))
        return A, B


def show_analysis(data, param, res):
    """ Display the results of the morphodynamics analysis. """

    # if param.showCircularity:
    #     pp = PdfPages(param.resultdir + "Circularity.pdf")
    #     fh.open_figure('Length', 1, (16, 9))
    #     plt.plot(res.length)
    #     pp.savefig()
    #     fh.open_figure('Area', 2, (16, 9))
    #     plt.plot(res.area)
    #     pp.savefig()
    #     fh.open_figure('Circularity: Length^2 / Area / 4 / pi', 2, (16, 9))
    #     plt.plot(res.length**2 / res.area / 4 / math.pi)
    #     pp.savefig()
    #     pp.close()

    if param.showCircularity:
        show_circularity(param, data, res, export=True)

    if param.showEdgeOverview:
        show_edge_overview(param, data, res, export=True)

    if param.showEdgeVectorial:
        ev = EdgeVectorial(param, data, res)
        ev.show_edge_vectorial(param, data, res, curvature=False)

    if param.showEdgeRasterized:
        show_edge_rasterized(param, data, res)
        # show_edge_rasterized(param, data, res, mode='simple')
        # show_edge_rasterized(param, data, res, mode='cumulative')
        # show_edge_rasterized(param, data, res, mode='curvature')

    if param.showCurvature:
        show_curvature(param, data, res)

    if param.showDisplacement:
        show_displacement(param, res)

    if param.showSignals:
        show_signals(param, data, res, 'Mean')
        show_signals(param, data, res, 'Variance')

    if param.showCorrelation:
        show_correlation(param, data, res)

    if param.showFourierDescriptors:
        show_fourier_descriptors(param, data, res)

class BatchExport():
    def __init__(self, param, data, res):
        self.param = param
        self.data = data
        self.res = res

    def create_interface(self):
        out = ipw.Output()

        circularity_checkbox = ipw.Checkbox(
            value=self.param.showCircularity,
            description='Circularity',
            disabled=False,
            indent=False
        )

        edge_overview_checkbox = ipw.Checkbox(
            value=self.param.showEdgeOverview,
            description='Edge overview',
            disabled=False,
            indent=False
        )

        edge_vectorial_checkbox = ipw.Checkbox(
            value=self.param.showEdgeVectorial,
            description='Edge vectorial',
            disabled=False,
            indent=False
        )

        edge_rasterized_checkbox = ipw.Checkbox(
            value=self.param.showEdgeRasterized,
            description='Edge rasterized',
            disabled=False,
            indent=False
        )

        curvature_checkbox = ipw.Checkbox(
            value=self.param.showCurvature,
            description='Curvature',
            disabled=False,
            indent=False
        )

        displacement_checkbox = ipw.Checkbox(
            value=self.param.showDisplacement,
            description='Displacement',
            disabled=False,
            indent=False
        )

        signals_checkbox = ipw.Checkbox(
            value=self.param.showSignals,
            description='Signals',
            disabled=False,
            indent=False
        )

        correlation_checkbox = ipw.Checkbox(
            value=self.param.showCorrelation,
            description='Correlation',
            disabled=False,
            indent=False
        )

        fourier_descriptors_checkbox = ipw.Checkbox(
            value=self.param.showFourierDescriptors,
            description='Fourier descriptors',
            disabled=False,
            indent=False
        )

        export_button = ipw.Button(
            description='Export figures',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            # tooltip='Click me',
            # icon='check'  # (FontAwesome names without the `fa-` prefix)
        )

        def export_figures(change):
            with out:
                self.param.showCircularity = circularity_checkbox.value
                self.param.showEdgeOverview = edge_overview_checkbox.value
                self.param.showEdgeVectorial = edge_vectorial_checkbox.value
                self.param.showEdgeRasterized = edge_rasterized_checkbox.value
                self.param.showCurvature = curvature_checkbox.value
                self.param.showDisplacement = displacement_checkbox.value
                self.param.showSignals = signals_checkbox.value
                self.param.showCorrelation = correlation_checkbox.value
                self.param.showFourierDescriptors = fourier_descriptors_checkbox.value
                matplotlib.use('PDF')
                show_analysis(self.data, self.param, self.res)
                matplotlib.use('nbAgg')

        export_button.on_click(export_figures)

        display(circularity_checkbox, edge_overview_checkbox, edge_vectorial_checkbox, edge_rasterized_checkbox, curvature_checkbox, displacement_checkbox, signals_checkbox, correlation_checkbox, fourier_descriptors_checkbox, export_button)

        display(out)
