import dill
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.external.tifffile import imread
from ArtifactGeneration import FigureHelper
from DisplacementEstimation import showEdge
from Windowing import showWindows

def showSignals():
   """ Display the signals extracted from the cell. """

   fh = FigureHelper(not True)

   data = dill.load(open(fh.path + "Signals.pkl", "rb"))

   pp = PdfPages(fh.path + "Edge.pdf")
   K = len(data['spline'])
   dmax = np.max(np.abs(data['displacement']))
   for k in range(K-1):
      fh.openFigure('Frame ' + str(k) + ' to frame ' + str(k+1), 1, (12, 9))
      plt.imshow(imread(data['path'] + data['morphosrc'] + str(k + 1) + '.tif'), cmap='gray')
      # showWindows(w, find_boundaries(labelWindows(w0)))  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
      showEdge(data['spline'][k], data['spline'][k+1], data['param0'][k], data['param'][k], data['displacement'][:, k], dmax)  # Show edge structures (spline curves, displacement vectors, sampling windows)
      # plt.savefig('Edge ' + str(k) + '.tif')
      pp.savefig()
      fh.show()
   pp.close()

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

# morphosrc = 'w16TIRF-CFP\\RhoA_OP_his_02_w16TIRF-CFP_t'
# showSignals()