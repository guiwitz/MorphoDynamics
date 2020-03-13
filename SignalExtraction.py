import dill
import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ArtifactGenerator import Plot

def signalExtraction():
   # print(matplotlib.get_configdir())
   # print(matplotlib.matplotlib_fname())
   # quit()

   plot = Plot(True)

   # data = pickle.load(open(plot.path + "Signals.pkl", "rb"))
   data = dill.load(open(plot.path + "Signals.pkl", "rb"))

   pdf = PdfPages(plot.path + "Signals.pdf")

   cmax = np.max(np.abs(data['displacement']))

   plot.plotopen('Displacement', 1)
   plt.imshow(data['displacement'], cmap='bwr')
   plt.axis('auto')
   plt.xlabel('Frame index')
   plt.ylabel('Window index')
   plt.colorbar(label='Displacement [pixels]')
   plt.clim(-cmax, cmax)
   # plt.xticks(range(0, velocity.shape[1], 5))
   plot.plotclose(False)
   pdf.savefig()

   # plt.show()

   for m in range(len(data['sigsrc'])):
      for j in range(data['signal'].shape[2]):
         plot.plotopen('Signal: ' + data['sigsrc'][m](0)[0:25] + ' - Layer: ' + str(j), 1)
         plt.imshow(np.transpose(data['signal'][:, m, j, 0:int(48/2**j)]), cmap='plasma')
         plt.colorbar(label='Signal')
         plt.axis('auto')
         plt.xlabel('Frame index')
         plt.ylabel('Window index')
         # plt.xticks(range(0, signal.shape[1], 5))
         plot.plotclose(False)
         pdf.savefig()

   pdf.close()

# morphosrc = 'w16TIRF-CFP\\RhoA_OP_his_02_w16TIRF-CFP_t'
# signalExtraction()