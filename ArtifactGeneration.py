import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.external.tifffile import imsave
from scipy.interpolate import splev
from Correlation import show_correlation, correlate_arrays, get_range
from DisplacementEstimation import show_edge_scatter, show_edge_line, show_edge_image, compute_curvature
from FigureHelper import FigureHelper
from Settings import Struct


def show_circularity(param, res, size=(16, 9)):
    pp = PdfPages(param.resultdir + 'Circularity.pdf')

    plt.figure(figsize=size)
    plt.gca().set_title('Length')
    plt.plot(res.length)
    plt.tight_layout()
    pp.savefig()

    plt.figure(figsize=size)
    plt.gca().set_title('Area')
    plt.plot(res.area)
    plt.tight_layout()
    pp.savefig()

    plt.figure(figsize=size)
    plt.gca().set_title('Circularity = Length^2 / Area / 4 / pi')
    plt.plot(res.length ** 2 / res.area / 4 / math.pi)
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
    show_edge_scatter(res.spline[k], res.spline[k + 1], res.param0[k], res.param[k], f)  # Show edge structures (spline curves, displacement vectors, sampling windows)
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


# def show_circularity(output, res):
#     fh = FigureHelper('Circularity', output)
#
#     fh.open_figure('Length')
#     plt.plot(res.length)
#     fh.save_pdf()
#
#     fh.open_figure('Area')
#     plt.plot(res.area)
#     fh.save_pdf()
#
#     fh.open_figure('Circularity = Length^2 / Area / 4 / pi')
#     plt.plot(res.length ** 2 / res.area / 4 / math.pi)
#     fh.save_pdf()
#
#     fh.close()


# def show_edge_overview(output, data, res):
#     fh = FigureHelper('Edge overview', output)
#
#     fh.open_figure('Edge overview')  # , 1, (12, 9)
#     plt.imshow(data.load_frame_morpho(0), cmap='gray')
#     show_edge_line(res.spline)
#     fh.save_pdf()
#
#     fh.close()


# def show_edge_vectorial(output, data, res, frame=None, curvature=False):
#     if frame is None:
#         frame_range = range(data.K - 1)
#     else:
#         frame_range = [frame]
#
#     if curvature:
#         name = 'Edge animation with curvature'
#     else:
#         name = 'Edge animation'
#
#     fh = FigureHelper(name, output)
#     # dmax = np.max(np.abs(res.displacement))
#     for k in frame_range:
#         print(k)
#         fh.open_figure('Frame ' + str(k) + ' to frame ' + str(k + 1), 1, (12, 9))
#         plt.imshow(data.load_frame_morpho(k), cmap='gray')
#         # showWindows(w, find_boundaries(labelWindows(w0)))  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
#         if curvature:
#             f = compute_curvature(res.spline[k])
#         else:
#             f = res.displacement[:, k]
#         show_edge_scatter(res.spline[k], res.spline[k + 1], res.param0[k], res.param[k], f)  # Show edge structures (spline curves, displacement vectors, sampling windows)
#         fh.save_pdf()
#         # if k < 20:
#         #    plt.savefig(param.resultdir + 'Edge ' + str(k) + '.tif')
#         if k != frame_range[-1]:
#             fh.show()
#     fh.close()


def show_edge_rasterized(output, param, data, res, cumulative=False, frame=None):
    d = res.displacement
    if cumulative:
        d = np.cumsum(d, axis=1)

    if frame is None:
        frame_range = range(data.K - 1)
    else:
        frame_range = [frame]

    if param.edgeNormalization == 'global':
        dmax = np.max(np.abs(d))
    else:
        dmax = None

    fh = FigureHelper('Edge animation', output)
    fh.open_figure('Edge animation')
    for k in frame_range:
        x = show_edge_image(data.shape, res.spline[k], res.param0[k], d[:, k], 3, dmax)
        plt.imshow(x)
        fh.save_tiff(x)
        if k != frame_range[-1]:
            fh.show()
    fh.close()


def show_curvature(output, data, res):
    curvature = np.zeros((10001, data.K))
    for k in range(data.K):
        curvature[:, k] = compute_curvature(res.spline[k])

    fh = FigureHelper('Curvature', output)
    fh.open_figure('Curvature', 1)
    cmax = np.max(np.abs(curvature))
    plt.imshow(curvature, cmap='seismic', vmin=-cmax, vmax=cmax)
    plt.colorbar(label='Curvature')
    plt.axis('auto')
    plt.xlabel('Frame index')
    plt.ylabel('Position on contour')
    # imsave(param.resultdir + 'Curvature.tif', curvature.astype(np.float32), compress=6)
    fh.save_pdf()
    fh.close()


def show_displacement(output, param, res):
    fh = FigureHelper('Displacement', output)
    fh.open_figure('Displacement')
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
    fh.save_pdf()
    fh.save_tiff(res.displacement.astype(np.float32))

    dcum = np.cumsum(res.displacement, axis=1)

    fh.open_figure('Cumulative displacement')
    plt.imshow(dcum, cmap='seismic')
    plt.axis('auto')
    plt.xlabel('Frame index')
    plt.ylabel('Window index')
    plt.colorbar(label='Displacement [pixels]')
    cmax = np.max(np.abs(dcum))
    plt.clim(-cmax, cmax)
    # plt.xticks(range(0, velocity.shape[1], 5))
    fh.save_pdf()
    fh.close()


def show_signals(output, param, data, res):
    fh = FigureHelper('Signal mean', output)
    for m in range(len(data.signalfile)):
        for j in range(res.mean.shape[1]):
            fh.open_figure('Signal: ' + data.get_channel_name(m) + ' - Layer: ' + str(j), 1)
            if hasattr(param, 'scaling_mean') & (j == 0):
                plt.imshow(res.mean[m, j, 0:param.I // 2 ** j, :], cmap='jet', vmin=param.scaling_mean[m][0], vmax=param.scaling_mean[m][1])
            else:
                plt.imshow(res.mean[m, j, 0:param.I // 2 ** j, :], cmap='jet')
            plt.colorbar(label='Mean')
            plt.axis('auto')
            plt.xlabel('Frame index')
            plt.ylabel('Window index')
            # plt.xticks(range(0, mean.shape[1], 5))
            if j == 0:
                imsave(param.resultdir + 'Signal mean ' + data.get_channel_name(m) + '.tif', res.mean[m, j, 0:param.I // 2 ** j, :].astype(np.float32), compress=6)
            fh.save_pdf()
    fh.close()

    fh = FigureHelper('Signal variance', output)
    # tw = TiffWriter(param.resultdir + 'Variance.tif')
    for m in range(len(data.signalfile)):
        for j in range(res.mean.shape[1]):
            fh.open_figure('Signal: ' + data.get_channel_name(m) + ' - Layer: ' + str(j), 1)
            plt.imshow(res.var[m, j, 0:param.I // 2 ** j, :], cmap='jet')
            plt.colorbar(label='Variance')
            plt.axis('auto')
            plt.xlabel('Frame index')
            plt.ylabel('Window index')
            # plt.xticks(range(0, mean.shape[1], 5))
            if j == 0:
                imsave(param.resultdir + 'Signal variance ' + data.get_channel_name(m) + '.tif', res.var[m, j, 0:param.I // 2 ** j, :].astype(np.float32), compress=6)
            # if j == 0:
            #     tw.save(res.var[m, j, 0:param.I//2**j, :], compress=6)
            fh.save_pdf()
    fh.close()
    # tw.close()


def show_fourier_descriptors(output, data, res):
    fh = FigureHelper('Fourier descriptors', output)
    N = 1000
    for k in range(data.K):
        c = splev(np.linspace(0, 1, N, endpoint=False), res.spline[k])
        c = c[0] + 1j * c[1]
        chat = np.fft.fftshift(np.fft.fft(c))
        fh.open_figure('Fourier descriptors at frame ' + str(k), 1)
        plt.plot(range(-N // 2, N // 2), np.log(np.abs(chat)))
        plt.ylim(-5, 15)
        plt.xlabel('Frequency')
        plt.ylabel('log(abs(Fourier coefficients))')
        fh.save_pdf()
    fh.close()


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
        # show_circularity(output, res)
        show_circularity(param, res)

    if param.showEdgeOverview:
        # show_edge_overview(output, data, res)
        show_edge_overview(param, data, res)

    if param.showEdgeVectorial:
        # show_edge_vectorial(output, data, res)
        show_edge_vectorial(param, data, res, curvature=False)

    if param.showEdgeRasterized:
        show_edge_rasterized(output, param, data, res, False)

    if param.showCurvature:
        show_curvature(output, data, res)

    if param.showDisplacement:
        show_displacement(output, data, res)

    if param.showSignals:
        show_signals(output, param, data, res)

    if param.showCorrelation:
        fh = FigureHelper('Correlation', output)
        dcum = np.cumsum(res.displacement, axis=1)

        pp = PdfPages(param.resultdir + "Correlation mean.pdf")
        c = correlate_arrays(res.displacement, res.displacement, 'Pearson')
        show_correlation(fh, c, res.displacement, res.displacement, 'displacement', 'displacement', 'Pearson')
        pp.savefig()
        # show_average_correlation(fh, c, res.displacement, res.displacement)
        # pp.savefig()
        for m in range(len(data.signalfile)):
            for normalization in ['Pearson']:  # [None, 'unbiased', 'Pearson', 'Pearson-unbiased']:
                c = correlate_arrays(res.displacement, res.mean[m, 0], normalization)
                show_correlation(fh, c, res.displacement, res.mean[m, 0], 'displacement', data.get_channel_name(m), normalization)
                pp.savefig()
                # show_average_correlation(fh, c, res.displacement, res.mean[m, 0])
                # pp.savefig()
                c = correlate_arrays(dcum, res.mean[m, 0], normalization)
                show_correlation(fh, c, dcum, res.mean[m, 0], 'cumulative displacement', data.get_channel_name(m), normalization)
                pp.savefig()
                # show_average_correlation(fh, c, dcum, res.mean[m, 0])
                # pp.savefig()
        for m in range(len(data.signalfile)):
            for mprm in range(m+1, len(data.signalfile)):
                for normalization in ['Pearson']:  # [None, 'unbiased', 'Pearson', 'Pearson-unbiased']:
                    c = correlate_arrays(res.mean[m, 0], res.mean[mprm, 0], normalization)
                    show_correlation(fh, c, res.mean[m, 0], res.mean[mprm, 0], data.get_channel_name(m), data.get_channel_name(mprm), normalization)
                    pp.savefig()
                    # show_average_correlation(fh, c, res.mean[0, 0], res.mean[-1, 0])
                    # pp.savefig()
        pp.close()

        pp = PdfPages(param.resultdir + "Correlation variance.pdf")
        # c = correlate_arrays(res.displacement, res.displacement, 'Pearson')
        # show_correlation(fh, c, res.displacement, res.displacement, 'displacement', 'displacement', 'Pearson')
        # pp.savefig()
        # show_average_correlation(fh, c, res.displacement, res.displacement)
        # pp.savefig()
        for m in range(len(data.signalfile)):
            for normalization in ['Pearson']:  # [None, 'unbiased', 'Pearson', 'Pearson-unbiased']:
                c = correlate_arrays(res.displacement, res.var[m, 0], normalization)
                show_correlation(fh, c, res.displacement, res.var[m, 0], 'displacement', data.get_channel_name(m), normalization)
                pp.savefig()
                # show_average_correlation(fh, c, res.displacement, res.var[m, 0])
                # pp.savefig()
                c = correlate_arrays(dcum, res.var[m, 0], normalization)
                show_correlation(fh, c, dcum, res.var[m, 0], 'cumulative displacement', data.get_channel_name(m), normalization)
                pp.savefig()
                # show_average_correlation(fh, c, dcum, res.var[m, 0])
                # pp.savefig()
        for m in range(len(data.signalfile)):
            for mprm in range(m+1, len(data.signalfile)):
                for normalization in ['Pearson']:  # [None, 'unbiased', 'Pearson', 'Pearson-unbiased']:
                    c = correlate_arrays(res.var[m, 0], res.var[mprm, 0], normalization)
                    show_correlation(fh, c, res.var[m, 0], res.var[mprm, 0], data.get_channel_name(m), data.get_channel_name(mprm), normalization)
                    pp.savefig()
                    # show_average_correlation(fh, c, res.var[0, 0], res.var[-1, 0])
                    # pp.savefig()
        pp.close()

        pp = PdfPages(param.resultdir + "Correlation average.pdf")
        fh.open_figure('Average cross-correlation between displacement and signals at layer ' + str(0), 1)
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
        fh.open_figure('Average cross-correlation between displacement and signals at layer ' + str(0) + ' - Windows 40 to 59', 1)
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
        fh.open_figure('Average cross-correlation between cumulative displacement and signals at layer ' + str(0), 1)
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
        fh.open_figure('Average autocorrelation of displacement', 1)
        t = get_range(res.displacement.shape[1], res.displacement.shape[1])
        c = correlate_arrays(res.displacement, res.displacement, 'Pearson')
        plt.plot(t, np.mean(c, axis=0), color[n])
        plt.grid()
        plt.xlabel('Time lag [frames]')
        plt.ylabel('Correlation')
        pp.savefig()
        fh.open_figure('Average autocorrelation of cumulative displacement', 1)
        t = get_range(dcum.shape[1], dcum.shape[1])
        c = correlate_arrays(dcum, dcum, 'Pearson')
        plt.plot(t, np.mean(c, axis=0), color[n])
        plt.grid()
        plt.xlabel('Time lag [frames]')
        plt.ylabel('Correlation')
        pp.savefig()
        pp.close()

    if param.showFourierDescriptors:
        show_fourier_descriptors(output, data, res)
