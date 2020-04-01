import math
import dill
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.external.tifffile import imread, TiffWriter
from ArtifactGeneration import FigureHelper
from Correlation import show_correlation, correlate_arrays, get_range
from DisplacementEstimation import show_edge_scatter, show_edge_line, show_edge_image


def trim(s):
    return s.split('/')[0]


def show_data(path):
    """ Display the signals extracted from the cell. """

    class Config:
        showCircularity = True
        showEdge = not True
        showEdgePDF = not True
        showDisplacement = not True
        showSignals = not True
        showCorrelation = True
        # edgeNormalization = 'global'
        edgeNormalization = 'frame-by-frame'

    fh = FigureHelper(not True)

    data = dill.load(open(path + "/Data.pkl", "rb"))
    x = imread('C:/Work/UniBE2/Data/' + data['expdir'] + data['morphodir'] + '1.tif')
    dcum = np.cumsum(data['displacement'], axis=1)

    K = len(data['spline'])
    # K = 10
    # I = data['displacement'].shape[0]

    if Config.showCircularity:
        pp = PdfPages(fh.path + "Circularity.pdf")
        fh.open_figure('Length', 1, (16, 9))
        plt.plot(data['length'])
        pp.savefig()
        fh.open_figure('Area', 2, (16, 9))
        plt.plot(data['area'])
        pp.savefig()
        fh.open_figure('Circularity: Length^2 / Area / 4 / pi', 2, (16, 9))
        plt.plot(data['length']**2 / data['area'] / 4 / math.pi)
        pp.savefig()
        pp.close()

    if Config.showEdge:
        pp = PdfPages(fh.path + "Edge overview.pdf")
        fh.open_figure('Edges', 1, (12, 9))
        plt.imshow(x, cmap='gray')
        show_edge_line(data['spline'])
        pp.savefig()
        fh.show()
        pp.close()

        if Config.showEdgePDF:
            pp = PdfPages(fh.path + "Edge animation.pdf")
            # dmax = np.max(np.abs(data['displacement']))
            for k in range(K-1):
                print(k)
                fh.open_figure('Frame ' + str(k) + ' to frame ' + str(k+1), 1, (12, 9))
                plt.imshow(imread(data['path'] + data['morphosrc'] + str(k + 1) + '.tif'), cmap='gray')
                # showWindows(w, find_boundaries(labelWindows(w0)))  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
                show_edge_scatter(data['spline'][k], data['spline'][k + 1], data['param0'][k], data['param'][k], data['displacement'][:, k]) # Show edge structures (spline curves, displacement vectors, sampling windows)
                pp.savefig()
                # if k < 20:
                #    plt.savefig('Edge ' + str(k) + '.tif')
                fh.show()
            pp.close()

        if Config.edgeNormalization == 'global':
            dmax = np.max(np.abs(data['displacement']))
        else:
            dmax = None
        tw = TiffWriter('Edge animation.tif')
        for k in range(K - 1):
            tw.save(show_edge_image(x.shape, data['spline'][k], data['param0'][k], data['displacement'][:, k], 3, dmax), compress=6)
        tw.close()

        if Config.edgeNormalization == 'global':
            dmax = np.max(np.abs(dcum))
        else:
            dmax = None
        tw = TiffWriter('Edge animation with cumulative displacement.tif')
        for k in range(K - 1):
            tw.save(show_edge_image(x.shape, data['spline'][k], data['param0'][k], dcum[:, k], 3, dmax), compress=6)
        tw.close()

    if Config.showDisplacement:
        pp = PdfPages(fh.path + "Displacement.pdf")
        fh.open_figure('Displacement', 1)
        plt.imshow(data['displacement'], cmap='bwr')
        plt.axis('auto')
        plt.xlabel('Frame index')
        plt.ylabel('Window index')
        plt.colorbar(label='Displacement [pixels]')
        cmax = np.max(np.abs(data['displacement']))
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

    if Config.showSignals:
        pp = PdfPages(fh.path + "Signals.pdf")
        for m in range(len(data['sigdir'])):
            for j in range(data['signal'].shape[1]):
                fh.open_figure('Signal: ' + trim(data['sigdir'][m](0)) + ' - Layer: ' + str(j), 1)
                plt.imshow(data['signal'][m, j, 0:int(48/2**j), :], cmap='plasma')
                plt.colorbar(label='Signal')
                plt.axis('auto')
                plt.xlabel('Frame index')
                plt.ylabel('Window index')
                # plt.xticks(range(0, signal.shape[1], 5))
                pp.savefig()
                fh.show()
                fh.close_figure()
        pp.close()

    if Config.showCorrelation:
        pp = PdfPages(fh.path + "Correlation.pdf")
        c = correlate_arrays(data['displacement'], data['displacement'], 'Pearson')
        show_correlation(fh, c, data['displacement'], data['displacement'], 'displacement', 'displacement', 'Pearson')
        pp.savefig()
        # show_average_correlation(fh, c, data['displacement'], data['displacement'])
        # pp.savefig()
        for m in range(len(data['sigdir'])):
            for normalization in ['Pearson']:  # [None, 'unbiased', 'Pearson', 'Pearson-unbiased']:
                c = correlate_arrays(data['displacement'], data['signal'][m, 0], normalization)
                show_correlation(fh, c, data['displacement'], data['signal'][m, 0], 'displacement', trim(data['sigdir'][m](0)), normalization)
                pp.savefig()
                # show_average_correlation(fh, c, data['displacement'], data['signal'][m, 0])
                # pp.savefig()
                c = correlate_arrays(dcum, data['signal'][m, 0], normalization)
                show_correlation(fh, c, dcum, data['signal'][m, 0], 'cumulative displacement', trim(data['sigdir'][m](0)), normalization)
                pp.savefig()
                # show_average_correlation(fh, c, dcum, data['signal'][m, 0])
                # pp.savefig()
        for normalization in ['Pearson']:  # [None, 'unbiased', 'Pearson', 'Pearson-unbiased']:
            c = correlate_arrays(data['signal'][0, 0], data['signal'][-1, 0], normalization)
            show_correlation(fh, c, data['signal'][0, 0], data['signal'][-1, 0], trim(data['sigdir'][0](0)), trim(data['sigdir'][-1](0)), normalization)
            pp.savefig()
            # show_average_correlation(fh, c, data['signal'][0, 0], data['signal'][-1, 0])
            # pp.savefig()
        pp.close()

        pp = PdfPages(fh.path + "Correlation comparison.pdf")
        fh.open_figure('Average cross-correlation between displacement and signals at layer ' + str(0), 1)
        color = 'rbgymc'
        n = 0
        for m in [0, len(data['sigdir'])-1]:
            t = get_range(data['displacement'].shape[1], data['signal'][m, 0].shape[1])
            c = correlate_arrays(data['displacement'], data['signal'][m, 0], 'Pearson')
            plt.plot(t, np.mean(c, axis=0), color[n], label=trim(data['sigdir'][m](0)))
            n += 1
        plt.grid()
        plt.legend(loc="upper left")
        plt.xlabel('Time lag [frames]')
        plt.ylabel('Cross-correlation')
        pp.savefig()
        fh.open_figure('Average cross-correlation between cumulative displacement and signals at layer ' + str(0), 1)
        color = 'rbgymc'
        n = 0
        for m in [0, len(data['sigdir'])-1]:
            t = get_range(dcum.shape[1], data['signal'][m, 0].shape[1])
            c = correlate_arrays(dcum, data['signal'][m, 0], 'Pearson')
            plt.plot(t, np.mean(c, axis=0), color[n], label=trim(data['sigdir'][m](0)))
            n += 1
        plt.grid()
        plt.legend(loc="upper left")
        plt.xlabel('Time lag [frames]')
        plt.ylabel('Cross-correlation')
        pp.savefig()
        fh.open_figure('Autocorrelation of displacement', 1)
        t = get_range(data['displacement'].shape[1], data['displacement'].shape[1])
        c = correlate_arrays(data['displacement'], data['displacement'], 'Pearson')
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


# path = 'Synthetic data'
# path = 'FRET_sensors + actinHistamineExpt2'
# path = 'FRET_sensors + actinPDGFRhoA_multipoint_0.5fn_s3_good'
path = 'GBD_sensors + actinExpt_01'
show_data(path)
