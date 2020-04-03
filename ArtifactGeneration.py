import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.external.tifffile import TiffWriter
from Correlation import show_correlation, correlate_arrays, get_range
from DisplacementEstimation import show_edge_scatter, show_edge_line, show_edge_image
from FigureHelper import FigureHelper


# def trim(s):
#     return s.split('/')[0]


def show_analysis(ds, config, res):
    """ Display the results of the morphodynamics analysis. """

    datadir = 'C:/Work/UniBE2/Data/'
    resultdir = ds.name + '/'
    x = ds.load_frame_morpho(0)

    dcum = np.cumsum(res.displacement, axis=1)
    fh = FigureHelper(not True)

    K = ds.K
    # I = res.displacement.shape[0]

    if config.showCircularity:
        pp = PdfPages(resultdir + "Circularity.pdf")
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

    if config.showEdge:
        pp = PdfPages(resultdir + "Edge overview.pdf")
        fh.open_figure('Edges', 1, (12, 9))
        plt.imshow(x, cmap='gray')
        show_edge_line(res.spline)
        pp.savefig()
        fh.show()
        pp.close()

        if config.showEdgePDF:
            pp = PdfPages(resultdir + "Edge animation.pdf")
            # dmax = np.max(np.abs(res.displacement))
            for k in range(K-1):
                print(k)
                fh.open_figure('Frame ' + str(k) + ' to frame ' + str(k+1), 1, (12, 9))
                plt.imshow(ds.load_frame_morpho(k), cmap='gray')
                # showWindows(w, find_boundaries(labelWindows(w0)))  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
                show_edge_scatter(res.spline[k], res.spline[k + 1], res.param0[k], res.param[k], res.displacement[:, k])  # Show edge structures (spline curves, displacement vectors, sampling windows)
                pp.savefig()
                # if k < 20:
                #    plt.savefig(resultdir + 'Edge ' + str(k) + '.tif')
                fh.show()
            pp.close()

        if config.edgeNormalization == 'global':
            dmax = np.max(np.abs(res.displacement))
        else:
            dmax = None
        tw = TiffWriter(resultdir + 'Edge animation.tif')
        for k in range(K - 1):
            tw.save(show_edge_image(x.shape, res.spline[k], res.param0[k], res.displacement[:, k], 3, dmax), compress=6)
        tw.close()

        if config.edgeNormalization == 'global':
            dmax = np.max(np.abs(dcum))
        else:
            dmax = None
        tw = TiffWriter(resultdir + 'Edge animation with cumulative displacement.tif')
        for k in range(K - 1):
            tw.save(show_edge_image(x.shape, res.spline[k], res.param0[k], dcum[:, k], 3, dmax), compress=6)
        tw.close()

    if config.showDisplacement:
        pp = PdfPages(resultdir + "Displacement.pdf")
        fh.open_figure('Displacement', 1)
        plt.imshow(res.displacement, cmap='bwr')
        plt.axis('auto')
        plt.xlabel('Frame index')
        plt.ylabel('Window index')
        plt.colorbar(label='Displacement [pixels]')
        cmax = np.max(np.abs(res.displacement))
        plt.clim(-cmax, cmax)
        # plt.xticks(range(0, velocity.shape[1], 5))
        pp.savefig()
        fh.show()
        fh.close_figure()

        fh.open_figure('Cumulative displacement', 1)
        plt.imshow(dcum, cmap='bwr')
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

    if config.showSignals:
        pp = PdfPages(resultdir + "Signals.pdf")
        for m in range(len(ds.signalfile)):
            for j in range(res.signal.shape[1]):
                fh.open_figure('Signal: ' + ds.get_channel_name(m) + ' - Layer: ' + str(j), 1)
                plt.imshow(res.signal[m, j, 0:int(48/2**j), :], cmap='plasma')
                plt.colorbar(label='Signal')
                plt.axis('auto')
                plt.xlabel('Frame index')
                plt.ylabel('Window index')
                # plt.xticks(range(0, signal.shape[1], 5))
                pp.savefig()
                fh.show()
                fh.close_figure()
        pp.close()

    if config.showCorrelation:
        pp = PdfPages(resultdir + "Correlation.pdf")
        c = correlate_arrays(res.displacement, res.displacement, 'Pearson')
        show_correlation(fh, c, res.displacement, res.displacement, 'displacement', 'displacement', 'Pearson')
        pp.savefig()
        # show_average_correlation(fh, c, res.displacement, res.displacement)
        # pp.savefig()
        for m in range(len(ds.signalfile)):
            for normalization in ['Pearson']:  # [None, 'unbiased', 'Pearson', 'Pearson-unbiased']:
                c = correlate_arrays(res.displacement, res.signal[m, 0], normalization)
                show_correlation(fh, c, res.displacement, res.signal[m, 0], 'displacement', ds.get_channel_name(m), normalization)
                pp.savefig()
                # show_average_correlation(fh, c, res.displacement, res.signal[m, 0])
                # pp.savefig()
                c = correlate_arrays(dcum, res.signal[m, 0], normalization)
                show_correlation(fh, c, dcum, res.signal[m, 0], 'cumulative displacement', ds.get_channel_name(m), normalization)
                pp.savefig()
                # show_average_correlation(fh, c, dcum, res.signal[m, 0])
                # pp.savefig()
        for normalization in ['Pearson']:  # [None, 'unbiased', 'Pearson', 'Pearson-unbiased']:
            c = correlate_arrays(res.signal[0, 0], res.signal[-1, 0], normalization)
            show_correlation(fh, c, res.signal[0, 0], res.signal[-1, 0], ds.get_channel_name(0), ds.get_channel_name(len(ds.signalfile)-1), normalization)
            pp.savefig()
            # show_average_correlation(fh, c, res.signal[0, 0], res.signal[-1, 0])
            # pp.savefig()
        pp.close()

        pp = PdfPages(resultdir + "Correlation comparison.pdf")
        fh.open_figure('Average cross-correlation between displacement and signals at layer ' + str(0), 1)
        color = 'rbgymc'
        n = 0
        for m in [0, len(ds.signalfile) - 1]:
            t = get_range(res.displacement.shape[1], res.signal[m, 0].shape[1])
            c = correlate_arrays(res.displacement, res.signal[m, 0], 'Pearson')
            plt.plot(t, np.mean(c, axis=0), color[n], label=ds.get_channel_name(m))
            n += 1
        plt.grid()
        plt.legend(loc="upper left")
        plt.xlabel('Time lag [frames]')
        plt.ylabel('Cross-correlation')
        pp.savefig()
        fh.open_figure('Average cross-correlation between cumulative displacement and signals at layer ' + str(0), 1)
        color = 'rbgymc'
        n = 0
        for m in [0, len(ds.signalfile) - 1]:
            t = get_range(dcum.shape[1], res.signal[m, 0].shape[1])
            c = correlate_arrays(dcum, res.signal[m, 0], 'Pearson')
            plt.plot(t, np.mean(c, axis=0), color[n], label=ds.get_channel_name(m))
            n += 1
        plt.grid()
        plt.legend(loc="upper left")
        plt.xlabel('Time lag [frames]')
        plt.ylabel('Cross-correlation')
        pp.savefig()
        fh.open_figure('Autocorrelation of displacement', 1)
        t = get_range(res.displacement.shape[1], res.displacement.shape[1])
        c = correlate_arrays(res.displacement, res.displacement, 'Pearson')
        plt.plot(t, np.mean(c, axis=0), color[n])
        plt.grid()
        plt.xlabel('Time lag [frames]')
        plt.ylabel('Correlation')
        pp.savefig()
        fh.open_figure('Autocorrelation of cumulative displacement', 1)
        t = get_range(dcum.shape[1], dcum.shape[1])
        c = correlate_arrays(dcum, dcum, 'Pearson')
        plt.plot(t, np.mean(c, axis=0), color[n])
        plt.grid()
        plt.xlabel('Time lag [frames]')
        plt.ylabel('Correlation')
        pp.savefig()
        pp.close()
