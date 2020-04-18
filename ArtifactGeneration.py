import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.external.tifffile import TiffWriter, imsave
from Correlation import show_correlation, correlate_arrays, get_range
from DisplacementEstimation import show_edge_scatter, show_edge_line, show_edge_image, compute_curvature
from FigureHelper import FigureHelper


def show_analysis(data, param, res):
    """ Display the results of the morphodynamics analysis. """

    x = data.load_frame_morpho(0)

    dcum = np.cumsum(res.displacement, axis=1)
    curvature = np.zeros((10001, data.K))
    for k in range(data.K):
        curvature[:, k] = compute_curvature(res.spline[k])
    fh = FigureHelper(not True)

    if param.showCircularity:
        pp = PdfPages(param.resultdir + "Circularity.pdf")
        fh.open_figure('Length', 1, (16, 9))
        plt.plot(res.length)
        pp.savefig()
        fh.open_figure('Area', 2, (16, 9))
        plt.plot(res.area)
        pp.savefig()
        fh.open_figure('Circularity: Length^2 / Area / 4 / pi', 2, (16, 9))
        plt.plot(res.length**2 / res.area / 4 / math.pi)
        pp.savefig()
        pp.close()

    if param.showEdge:
        pp = PdfPages(param.resultdir + "Edge overview.pdf")
        fh.open_figure('Edges', 1, (12, 9))
        plt.imshow(x, cmap='gray')
        show_edge_line(res.spline)
        pp.savefig()
        fh.show()
        pp.close()

        if param.showEdgePDF:
            pp = PdfPages(param.resultdir + "Edge animation.pdf")
            # dmax = np.max(np.abs(res.displacement))
            for k in range(data.K-1):
                print(k)
                fh.open_figure('Frame ' + str(k) + ' to frame ' + str(k+1), 1, (12, 9))
                plt.imshow(data.load_frame_morpho(k), cmap='gray')
                # showWindows(w, find_boundaries(labelWindows(w0)))  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
                show_edge_scatter(res.spline[k], res.spline[k + 1], res.param0[k], res.param[k], res.displacement[:, k])  # Show edge structures (spline curves, displacement vectors, sampling windows)
                pp.savefig()
                # if k < 20:
                #    plt.savefig(param.resultdir + 'Edge ' + str(k) + '.tif')
                fh.show()
            pp.close()

        if param.edgeNormalization == 'global':
            dmax = np.max(np.abs(res.displacement))
        else:
            dmax = None
        tw = TiffWriter(param.resultdir + 'Edge animation.tif')
        for k in range(data.K - 1):
            tw.save(show_edge_image(x.shape, res.spline[k], res.param0[k], res.displacement[:, k], 3, dmax), compress=6)
        tw.close()

        if param.edgeNormalization == 'global':
            dmax = np.max(np.abs(dcum))
        else:
            dmax = None
        tw = TiffWriter(param.resultdir + 'Edge animation with cumulative displacement.tif')
        for k in range(data.K - 1):
            tw.save(show_edge_image(x.shape, res.spline[k], res.param0[k], dcum[:, k], 3, dmax), compress=6)
        tw.close()

    if param.showCurvature:
        # pp = PdfPages(param.resultdir + "Edge animation with curvature.pdf")
        # # dmax = np.max(np.abs(res.displacement))
        # for k in range(data.K - 1):
        #     print(k)
        #     fh.open_figure('Frame ' + str(k) + ' to frame ' + str(k + 1), 1, (12, 9))
        #     plt.imshow(data.load_frame_morpho(k), cmap='gray')
        #     # showWindows(w, find_boundaries(labelWindows(w0)))  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
        #     show_edge_scatter(res.spline[k], res.spline[k + 1], res.param0[k], res.param[k], compute_curvature(res.spline[k]))  # Show edge structures (spline curves, displacement vectors, sampling windows)
        #     pp.savefig()
        #     # if k < 20:
        #     #    plt.savefig(param.resultdir + 'Edge ' + str(k) + '.tif')
        #     fh.show()
        # pp.close()

        pp = PdfPages(param.resultdir + "Curvature.pdf")
        fh.open_figure('Curvature', 1)
        cmax = np.max(np.abs(curvature))
        plt.imshow(curvature, cmap='bwr', vmin=-cmax, vmax=cmax)
        plt.colorbar(label='Curvature')
        plt.axis('auto')
        plt.xlabel('Frame index')
        plt.ylabel('Position on contour')
        # imsave(param.resultdir + 'Curvature.tif', curvature.astype(np.float32), compress=6)
        pp.savefig()
        fh.show()
        fh.close_figure()
        pp.close()

    if param.showDisplacement:
        pp = PdfPages(param.resultdir + "Displacement.pdf")
        fh.open_figure('Displacement', 1)
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
        fh.show()
        fh.close_figure()
        imsave(param.resultdir + 'Displacement.tif', res.displacement.astype(np.float32))

        fh.open_figure('Cumulative displacement', 1)
        plt.imshow(dcum, cmap='seismic')
        plt.axis('auto')
        plt.xlabel('Frame index')
        plt.ylabel('Window index')
        plt.colorbar(label='Displacement [pixels]')
        cmax = np.max(np.abs(dcum))
        plt.clim(-cmax, cmax)
        # plt.xticks(range(0, velocity.shape[1], 5))
        pp.savefig()
        fh.show()
        fh.close_figure()
        pp.close()

    if param.showSignals:
        pp = PdfPages(param.resultdir + "Signal mean.pdf")
        for m in range(len(data.signalfile)):
            for j in range(res.mean.shape[1]):
                fh.open_figure('Signal: ' + data.get_channel_name(m) + ' - Layer: ' + str(j), 1)
                if hasattr(param, 'scaling_mean') & (j == 0):
                    plt.imshow(res.mean[m, j, 0:param.I // 2 ** j, :], cmap='jet', vmin=param.scaling_mean[m][0], vmax=param.scaling_mean[m][1])
                else:
                    plt.imshow(res.mean[m, j, 0:param.I//2**j, :], cmap='jet')
                plt.colorbar(label='Mean')
                plt.axis('auto')
                plt.xlabel('Frame index')
                plt.ylabel('Window index')
                # plt.xticks(range(0, mean.shape[1], 5))
                if j == 0:
                    imsave(param.resultdir + 'Signal mean ' + data.get_channel_name(m) + '.tif', res.mean[m, j, 0:param.I//2**j, :].astype(np.float32), compress=6)
                pp.savefig()
                fh.show()
                fh.close_figure()
        pp.close()
        pp = PdfPages(param.resultdir + "Signal variance.pdf")
        # tw = TiffWriter(param.resultdir + 'Variance.tif')
        for m in range(len(data.signalfile)):
            for j in range(res.mean.shape[1]):
                fh.open_figure('Signal: ' + data.get_channel_name(m) + ' - Layer: ' + str(j), 1)
                plt.imshow(res.var[m, j, 0:param.I//2**j, :], cmap='jet')
                plt.colorbar(label='Variance')
                plt.axis('auto')
                plt.xlabel('Frame index')
                plt.ylabel('Window index')
                # plt.xticks(range(0, mean.shape[1], 5))
                if j == 0:
                    imsave(param.resultdir + 'Signal variance ' + data.get_channel_name(m) + '.tif', res.var[m, j, 0:param.I//2**j, :].astype(np.float32), compress=6)
                # if j == 0:
                #     tw.save(res.var[m, j, 0:param.I//2**j, :], compress=6)
                pp.savefig()
                fh.show()
                fh.close_figure()
        pp.close()
        # tw.close()

    if param.showCorrelation:
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
        from scipy.interpolate import splev
        pp = PdfPages(param.resultdir + "Fourier descriptors.pdf")
        N = 1000
        for k in range(data.K):
            c = splev(np.linspace(0, 1, N, endpoint=False), res.spline[k])
            c = c[0] + 1j*c[1]
            chat = np.fft.fftshift(np.fft.fft(c))
            fh.open_figure('Fourier descriptors at frame ' + str(k), 1)
            plt.plot(range(-N//2, N//2), np.log(np.abs(chat)))
            plt.ylim(-5, 15)
            plt.xlabel('Frequency')
            plt.ylabel('log(abs(Fourier coefficients))')
            fh.show()
            pp.savefig()
        pp.close()