import dill
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ArtifactGeneration import FigureHelper

def showSignals():
   """ Display the signals extracted from the cell. """

   fh = FigureHelper(not True)
   pp = PdfPages(fh.path + "Signals.pdf")

   data = dill.load(open(fh.path + "Signals.pkl", "rb"))

   cmax = np.max(np.abs(data['displacement']))

   fh.openFigure('Displacement', 1)
   plt.imshow(data['displacement'], cmap='bwr')
   plt.axis('auto')
   plt.xlabel('Frame index')
   plt.ylabel('Window index')
   plt.colorbar(label='Displacement [pixels]')
   plt.clim(-cmax, cmax)
   # plt.xticks(range(0, velocity.shape[1], 5))
   pp.savefig()
   fh.show()
   fh.closeFigure()

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
# signalExtraction()