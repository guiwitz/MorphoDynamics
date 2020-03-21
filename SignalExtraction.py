import dill
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.external.tifffile import imread
from ArtifactGeneration import FigureHelper
from DisplacementEstimation import showEdgeScatter, showEdgeLine
from Windowing import showWindows

def correlate(x, y, normalization=None, removemean=True):
   def npcorrelate(x, y):
      return np.correlate(x, y, mode='full')

   if removemean:
      x = x - np.mean(x)
      y = y - np.mean(y)
   c = npcorrelate(x, y)
   if normalization == 'unbiased':
      c /= npcorrelate(np.ones(x.shape), np.ones(y.shape))
   elif normalization == 'Pearson':
      c /= np.linalg.norm(x) / np.linalg.norm(y)
   elif normalization == 'Pearson-unbiased':
      e = np.sqrt(npcorrelate(x**2, np.ones(y.shape)) * npcorrelate(np.ones(x.shape), y**2))
      e[e == 0] = 1
      c /= e
   return c

# K = 100
# # Visualize position corresponding to zero lag
# x = np.zeros((K,))
# x[0] = 1
# y = np.zeros((K,))
# y[0] = 1
# plt.figure()
# plt.plot(correlate(x, y))
# # Verify that correlation of ones gives normalization coefficients for unbiased mode
# x = np.ones((K,))
# y = np.ones((K,))
# plt.figure()
# plt.plot(correlate(x, y, normalization=None, removemean=False))
# # Verify autocorrelation of a sine with Pearson-unbiased normalization
# x = np.sin(2*np.pi*np.array(range(K))/10)
# plt.figure()
# plt.plot(correlate(x, x, normalization='Pearson-unbiased'))
# plt.show()
# quit()

def getExtent(A, B, I):
   return (-B + 1 - 0.5, -B + 1 + A + B - 2 + 0.5, I - 1 + 0.5, -0.5)

def trim(s):
   return s.split('/')[0]

def showSignals():
   """ Display the signals extracted from the cell. """

   fh = FigureHelper(not True)

   data = dill.load(open(fh.path + "Data.pkl", "rb"))
   ds = np.cumsum(data['displacement'], axis=1)

   K = len(data['spline'])
   I = data['displacement'].shape[0]

   def showCorrelation(x, y, sx, sy, normalization):
      fh.openFigure('Correlation between ' + sx + ' and ' + sy + ' at layer ' + str(0) + ' - Normalization: ' + str(normalization), 1, (16, 9))
      c = np.zeros((I, 2*K-2))
      for i in range(I):
         c[i] = correlate(x[i], y[:, m, 0, i], normalization=normalization, removemean=True)
      plt.imshow(c, extent=getExtent(K, K-1, I), cmap='bwr') # , vmin=-1, vmax=1
      plt.axis('auto')
      plt.xlabel('Time lag [frames]')
      plt.ylabel('Window index')
      plt.colorbar(label='Normalized correlation')
      pp.savefig()
      fh.show()

   pp = PdfPages(fh.path + "Correlation.pdf")
   for m in range(len(data['sigsrc'])):
      for normalization in [None, 'unbiased', 'Pearson', 'Pearson-unbiased']:
         showCorrelation(data['displacement'], data['signal'], 'displacement', trim(data['sigsrc'][m](0)), normalization)
         # fh.openFigure('Correlation between displacement and ' + trim(data['sigsrc'][m](0)) + ' at layer ' + str(0), 1, (16, 9))
         # c = np.zeros((I, 2 * K - 2))
         # for i in range(I):
         #    c[i] = correlate(data['displacement'][i], data['signal'][:, m, 0, i], normalization='unbiased', removemean=True)
         # plt.imshow(c, extent=getExtent(K, K - 1, I), cmap='bwr')  # , vmin=-1, vmax=1
         # plt.axis('auto')
         # plt.xlabel('Time lag [frames]')
         # plt.ylabel('Window index')
         # plt.colorbar(label='Normalized correlation')
         # pp.savefig()
         # fh.show()
         showCorrelation(ds, data['signal'], 'cumulative displacement', trim(data['sigsrc'][m](0)), normalization)
         # fh.openFigure('Correlation between cumulative displacement and ' + trim(data['sigsrc'][m](0)) + ' at layer ' + str(0), 1, (16, 9))
         # c = np.zeros((I, 2 * K - 2))
         # for i in range(I):
         #    c[i] = correlate(ds[i], data['signal'][:, m, 0, i], normalization='unbiased', removemean=True)
         # plt.imshow(c, extent=getExtent(K, K - 1, I), cmap='bwr') # , vmin=-1, vmax=1
         # plt.axis('auto')
         # plt.xlabel('Time lag [frames]')
         # plt.ylabel('Window index')
         # plt.colorbar(label='Normalized correlation')
         # pp.savefig()
         # fh.show()
   # fh.openFigure('Correlation between ' + trim(data['sigsrc'][0](0)) + ' and ' + trim(data['sigsrc'][4](0)) + ' at layer ' + str(0), 1, (16, 9))
   # c = np.zeros((I, 2 * K - 1))
   # for i in range(I):
   #    c[i] = correlate(data['signal'][:, 0, 0, i], data['signal'][:, 4, 0, i], normalization='Pearson-unbiased')
   # plt.imshow(c, extent=getExtent(K, K, I), cmap='bwr', vmin=-1, vmax=1)
   # plt.axis('auto')
   # plt.xlabel('Time lag [frames]')
   # plt.ylabel('Window index')
   # plt.colorbar(label='Normalized correlation')
   # pp.savefig()
   # fh.show()
   pp.close()
   quit()

   pp = PdfPages(fh.path + "Edge overview.pdf")
   fh.openFigure('Edges', 1, (12, 9))
   plt.imshow(imread(data['path'] + data['morphosrc'] + '1.tif'), cmap='gray')
   showEdgeLine(data['spline'])
   pp.savefig()
   fh.show()
   pp.close()

   pp = PdfPages(fh.path + "Edge animation.pdf")
   dmax = np.max(np.abs(data['displacement']))
   for k in range(K-1):
      fh.openFigure('Frame ' + str(k) + ' to frame ' + str(k+1), 1, (12, 9))
      plt.imshow(imread(data['path'] + data['morphosrc'] + str(k + 1) + '.tif'), cmap='gray')
      # showWindows(w, find_boundaries(labelWindows(w0)))  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
      showEdgeScatter(data['spline'][k], data['spline'][k + 1], data['param0'][k], data['param'][k], data['displacement'][:, k], dmax)  # Show edge structures (spline curves, displacement vectors, sampling windows)
      plt.savefig('Edge ' + str(k) + '.tif')
      pp.savefig()
      fh.show()
   pp.close()
   # quit()

   pp = PdfPages(fh.path + "Displacement.pdf")
   fh.openFigure('Displacement', 1)
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
   fh.closeFigure()

   fh.openFigure('Cumulative displacement', 1)
   plt.imshow(ds, cmap='bwr')
   plt.axis('auto')
   plt.xlabel('Frame index')
   plt.ylabel('Window index')
   plt.colorbar(label='Displacement [pixels]')
   cmax = np.max(np.abs(ds))
   plt.clim(-cmax, cmax)
   # plt.xticks(range(0, velocity.shape[1], 5))
   pp.savefig()
   fh.show()
   fh.closeFigure()
   pp.close()

   pp = PdfPages(fh.path + "Signals.pdf")
   for m in range(len(data['sigsrc'])):
      for j in range(data['signal'].shape[2]):
         fh.openFigure('Signal: ' + data['sigsrc'][m](0)[0:25] + ' - Layer: ' + str(j), 1)
         plt.imshow(np.transpose(data['signal'][:, m, j, 0:int(48/2**j)]), cmap='plasma')
         plt.colorbar(label='Signal')
         plt.axis('auto')
         plt.xlabel('Frame index')
         plt.ylabel('Window index')
         # plt.xticks(range(0, signal.shape[1], 5))
         pp.savefig()
         fh.show()
         fh.closeFigure()
   pp.close()

showSignals()