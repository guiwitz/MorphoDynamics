import math
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as ipw
from IPython.display import display
from matplotlib.backends.backend_pdf import PdfPages
from skimage.external.tifffile import imsave, TiffWriter
from scipy.interpolate import splev
from PIL import Image
from Correlation import show_correlation_core, correlate_arrays, get_range
from DisplacementEstimation import show_edge_scatter, show_edge_line, show_edge_image, compute_curvature, compute_length, compute_area
from Settings import Struct


def show_circularity(param, data, res, size=(16, 9)):
    pp = PdfPages(param.resultdir + 'Circularity.pdf')

    length = np.zeros((data.K,))
    area = np.zeros((data.K,))
    for k in range(data.K):
        length[k] = compute_length(res.spline[k])  # Length of the contour
        area[k] = compute_area(res.spline[k])  # Area delimited by the contour

    plt.figure(figsize=size)
    plt.gca().set_title('Length')
    plt.plot(length)
    plt.tight_layout()
    pp.savefig()

    plt.figure(figsize=size)
    plt.gca().set_title('Area')
    plt.plot(area)
    plt.tight_layout()
    pp.savefig()

    plt.figure(figsize=size)
    plt.gca().set_title('Circularity = Length^2 / Area / 4 / pi')
    plt.plot(length ** 2 / area / 4 / math.pi)
    plt.tight_layout()
    pp.savefig()

    pp.close()


def show_edge_overview(param, data, res, size=(12,9)):
    pp = PdfPages(param.resultdir + 'Edge overview.pdf')

    plt.figure(figsize=size)
    plt.gca().set_title('Edge overview')
    plt.imshow(data.load_frame_morpho(0), cmap='gray')
    show_edge_line(res.spline)
    plt.tight_layout()
    pp.savefig()

    pp.close()


def show_edge_vectorial_aux(data, res, k, curvature=False):
    plt.clf()
    plt.gca().set_title('Frame ' + str(k) + ' to frame ' + str(k + 1))
    plt.imshow(data.load_frame_morpho(k), cmap='gray')
    # showWindows(w, find_boundaries(labelWindows(w0)))  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
    if curvature:
        f = compute_curvature(res.spline[k])
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
    pp = PdfPages(param.resultdir + name + '.pdf')

    plt.text(0.5, 0.5, 'This page intentionally left blank.')
    pp.savefig()

    # dmax = np.max(np.abs(res.displacement))
    for k in range(data.K - 1):
        print(k)
        show_edge_vectorial_aux(data, res, k, curvature)
        pp.savefig()
    pp.close()


def show_edge_rasterized_aux(data, res, d, dmax, k, mode, display=True):
    if mode == 'curvature':
        x = show_edge_image(data.shape, res.spline[k], np.linspace(0, 1, 10001), d[:, k], 3, dmax)
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
        t = np.linspace(0, 1, 10001)
        d = np.zeros((10001, data.K))
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

    tw = TiffWriter(param.resultdir + name + '.tif')
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
            options=['cumulative', 'simple', 'curvature', 'default'],
            value='default',
            #    layout={'width': 'max-content'}, # If the items' names are long
            description='Mode:'
            # disabled=False
        )

        def mode_change(change):
            with out:
                self.set_mode(change['new'])

        mode_selector.observe(mode_change, names='value')

        time_slider = ipw.IntSlider(description='Time', min=0, max=self.data.K - 1, continuous_update=False, layout=ipw.Layout(width='100%'))

        def time_change(change):
            with out:
                # show_edge_rasterized_aux(param, change['new'], cumulative=False)
                self.plot(change['new'])

        time_slider.observe(time_change, names='value')

        display(mode_selector)
        display(time_slider)
        display(out)

    def set_mode(self, mode):
        if mode == 'cumulative':
            self.d = np.cumsum(self.res.displacement, axis=1)
            self.name = 'Edge animation (cumulative)'
        elif mode == 'simple':
            self.d = np.ones(self.res.displacement.shape)
            self.name = 'Edge animation (simple)'
        elif mode == 'curvature':
            t = np.linspace(0, 1, 10000, endpoint=False)
            self.d = np.zeros((10000, self.data.K))
            for k in range(self.data.K):
                self.d[:, k] = compute_curvature(self.res.spline[k], t)
            self.name = 'Edge animation (curvature)'
        else:
            self.d = self.res.displacement
            self.name = 'Edge animation'

    def setNormalization(self, normalization):
        if normalization == 'global':
            self.dmax = np.max(np.abs(d))
        else:
            self.dmax = None

    def getImage(self, k):
        x = show_edge_rasterized_aux(self.data, self.res, self.d, self.dmax, k, self.mode, display=False)
        y0 = np.stack((self.y[k], self.y[k], self.y[k]), axis=-1)
        x[x == 0] = y0[x == 0]
        return x

    def save(self):
        tw = TiffWriter(self.param.resultdir + self.name + '.tif')
        for k in range(self.data.K - 1):
            tw.save(self.getImage(k), compress=6)
        tw.close()

    def plot(self, k):
        plt.clf()
        plt.gca().set_title('Frame ' + str(k))
        plt.imshow(self.getImage(k))
        plt.tight_layout()


def show_curvature(param, data, res, size=(16, 9)):
    curvature = np.zeros((10001, data.K))
    for k in range(data.K):
        curvature[:, k] = compute_curvature(res.spline[k], np.linspace(0, 1, 10001))

    plt.figure(figsize=size)
    plt.gca().set_title('Curvature')
    pp = PdfPages(param.resultdir + 'Curvature.pdf')
    cmax = np.max(np.abs(curvature))
    plt.imshow(curvature, cmap='seismic', vmin=-cmax, vmax=cmax)
    plt.colorbar(label='Curvature')
    plt.axis('auto')
    plt.xlabel('Frame index')
    plt.ylabel('Position on contour')
    # imsave(param.resultdir + 'Curvature.tif', curvature.astype(np.float32), compress=6)
    pp.savefig()
    pp.close()


def show_displacement(param, res, size=(16, 9)):
    pp = PdfPages(param.resultdir + 'Displacement.pdf')

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
    pp.savefig()

    pp.close()


def show_signals(param, data, res, size=(16, 9)):
    f = plt.figure(figsize=size)

    pp = PdfPages(param.resultdir + 'Signal mean.pdf')
    for m in range(len(data.signalfile)):
        for j in range(res.mean.shape[1]):
            plt.figure(f.number)
            plt.clf()
            plt.gca().set_title('Signal: ' + data.get_channel_name(m) + ' - Layer: ' + str(j))
            if hasattr(param, 'scaling_mean') & (j == 0):
                plt.imshow(res.mean[m, j, 0:res.I[j], :], cmap='jet', vmin=param.scaling_mean[m][0], vmax=param.scaling_mean[m][1])
            else:
                plt.imshow(res.mean[m, j, 0:res.I[j], :], cmap='jet')
            plt.colorbar(label='Mean')
            plt.axis('auto')
            plt.xlabel('Frame index')
            plt.ylabel('Window index')
            # plt.xticks(range(0, mean.shape[1], 5))
            if j == 0:
                imsave(param.resultdir + 'Signal mean ' + data.get_channel_name(m) + '.tif', res.mean[m, j, 0:res.I[j], :].astype(np.float32), compress=6)
            pp.savefig()
    pp.close()

    pp = PdfPages(param.resultdir + 'Signal variance.pdf')
    # tw = TiffWriter(param.resultdir + 'Variance.tif')
    for m in range(len(data.signalfile)):
        for j in range(res.mean.shape[1]):
            plt.figure(f.number)
            plt.clf()
            plt.gca().set_title('Signal: ' + data.get_channel_name(m) + ' - Layer: ' + str(j))
            plt.imshow(res.var[m, j, 0:res.I[j], :], cmap='jet')
            plt.colorbar(label='Variance')
            plt.axis('auto')
            plt.xlabel('Frame index')
            plt.ylabel('Window index')
            # plt.xticks(range(0, mean.shape[1], 5))
            if j == 0:
                imsave(param.resultdir + 'Signal variance ' + data.get_channel_name(m) + '.tif', res.var[m, j, 0:res.I[j], :].astype(np.float32), compress=6)
            # if j == 0:
            #     tw.save(res.var[m, j, 0:param.I//2**j, :], compress=6)
            pp.savefig()
    pp.close()
    # tw.close()


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
    pp = PdfPages(param.resultdir + 'Fourier descriptors.pdf')
    for k in range(data.K):
        show_fourier_descriptors_aux(res, k)
        pp.savefig()
    pp.close()

def show_correlation(param, data, res, size=(16, 9)):
    dcum = np.cumsum(res.displacement, axis=1)

    plt.figure(figsize=size)

    pp = PdfPages(param.resultdir + "Correlation mean.pdf")
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

    pp = PdfPages(param.resultdir + "Correlation variance.pdf")
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
            show_correlation_core(c, res.var[m, 0], res.var[mprm, 0], data.get_channel_name(m),
                             data.get_channel_name(mprm), 'Pearson')
            pp.savefig()
            # show_average_correlation(fh, c, res.var[0, 0], res.var[-1, 0])
            # pp.savefig()
    pp.close()

    pp = PdfPages(param.resultdir + "Correlation average.pdf")
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
    pp.savefig()

    plt.clf()
    plt.gca().set_title('Average cross-correlation between displacement and signals at layer ' + str(0) + ' - Windows 40 to 59')
    color = 'rbgymc'
    n = 0
    for m in [0, len(data.signalfile) - 1]:
        t = get_range(res.displacement.shape[1], res.mean[m, 0].shape[1])
        c = correlate_arrays(res.displacement[40:60], res.mean[m, 0][40:60], 'Pearson')
        plt.plot(t, np.mean(c, axis=0), color[n], label=data.get_channel_name(m))
        n += 1
    plt.grid()
    plt.legend(loc="upper left")
    plt.xlabel('Time lag [frames]')
    plt.ylabel('Cross-correlation')
    pp.savefig()

    plt.clf()
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
    pp.savefig()

    plt.clf()
    plt.gca().set_title('Average autocorrelation of displacement')
    t = get_range(res.displacement.shape[1], res.displacement.shape[1])
    c = correlate_arrays(res.displacement, res.displacement, 'Pearson')
    plt.plot(t, np.mean(c, axis=0), color[n])
    plt.grid()
    plt.xlabel('Time lag [frames]')
    plt.ylabel('Correlation')
    pp.savefig()

    plt.clf()
    plt.gca().set_title('Average autocorrelation of cumulative displacement')
    t = get_range(dcum.shape[1], dcum.shape[1])
    c = correlate_arrays(dcum, dcum, 'Pearson')
    plt.plot(t, np.mean(c, axis=0), color[n])
    plt.grid()
    plt.xlabel('Time lag [frames]')
    plt.ylabel('Correlation')
    pp.savefig()
    pp.close()


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

    output = Struct()
    output.dir = param.resultdir
    output.size = (16, 9)
    output.display = False
    output.pdf = True
    output.tiff = True

    if param.showCircularity:
        show_circularity(param, data, res)

    if param.showEdgeOverview:
        show_edge_overview(param, data, res)

    if param.showEdgeVectorial:
        show_edge_vectorial(param, data, res, curvature=False)

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
        show_signals(param, data, res)

    if param.showCorrelation:
        show_correlation(param, data, res)

    if param.showFourierDescriptors:
        show_fourier_descriptors(param, data, res)
