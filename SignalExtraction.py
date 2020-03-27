import dill
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.external.tifffile import imread, TiffWriter
from ArtifactGeneration import FigureHelper
from DisplacementEstimation import show_edge_scatter, show_edge_line, show_edge_image, show_edge_image, rasterize_curve
# from Windowing import showWindows


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
		c /= np.linalg.norm(x) * np.linalg.norm(y)
	elif normalization == 'Pearson-unbiased':
		e = np.sqrt(npcorrelate(x ** 2, np.ones(y.shape)) * npcorrelate(np.ones(x.shape), y ** 2))
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
	A, B = max(A, B), min(A, B)
	return -B + 1 - 0.5, -B + 1 + A + B - 2 + 0.5, I - 1 + 0.5, -0.5


def trim(s):
	return s.split('/')[0]


def showSignals(path):
	""" Display the signals extracted from the cell. """

	class Config:
		showEdge = not False
		showDisplacement = not False
		showSignals = not False
		showCorrelation = not False

	fh = FigureHelper(not True)

	data = dill.load(open(path + "/Data.pkl", "rb"))
	dcum = np.cumsum(data['displacement'], axis=1)

	K = len(data['spline'])
	K = 10
	I = data['displacement'].shape[0]

	if Config.showEdge:
		pp = PdfPages(fh.path + "1 Edge overview.pdf")
		fh.openFigure('Edges', 1, (12, 9))
		plt.imshow(imread(data['path'] + data['morphosrc'] + '1.tif'), cmap='gray')
		show_edge_line(data['spline'])
		pp.savefig()
		fh.show()
		pp.close()

		pp = PdfPages(fh.path + "2 Edge animation.pdf")
		dmax = np.max(np.abs(data['displacement']))
		tw = TiffWriter('Edge.tif')
		for k in range(K-1):
			print(k)
			fh.openFigure('Frame ' + str(k) + ' to frame ' + str(k+1), 1, (12, 9))
			x = imread(data['path'] + data['morphosrc'] + str(k + 1) + '.tif')
			plt.imshow(x, cmap='gray')
			# showWindows(w, find_boundaries(labelWindows(w0)))  # Show window boundaries and their indices; for a specific window, use: w0[0, 0].astype(dtype=np.uint8)
			# showEdgeScatter(data['spline'][k], data['spline'][k + 1], data['param0'][k], data['param'][k], data['displacement'][:, k], np.max(data['displacement'][:, k])) # Show edge structures (spline curves, displacement vectors, sampling windows)
			# pp.savefig()
			# if k < 20:
			#    plt.savefig('Edge ' + str(k) + '.tif')
			tw.save(show_edge_image(x.shape, data['spline'][k], data['param0'][k], data['displacement'][:, k], 1), compress=6)
			tw.save(show_edge_image(x.shape, data['spline'][k], data['param0'][k], data['displacement'][:, k], 3), compress=6)
			fh.show()
		tw.close()
		pp.close()

	if Config.showDisplacement:
		pp = PdfPages(fh.path + "3 Displacement.pdf")
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
		fh.closeFigure()
		pp.close()

	if Config.showSignals:
		pp = PdfPages(fh.path + "4 Signals.pdf")
		for m in range(len(data['sigsrc'])):
			for j in range(data['signal'].shape[1]):
				fh.openFigure('Signal: ' + trim(data['sigsrc'][m](0)) + ' - Layer: ' + str(j), 1)
				plt.imshow(data['signal'][m, j, 0:int(48/2**j), :], cmap='plasma')
				plt.colorbar(label='Signal')
				plt.axis('auto')
				plt.xlabel('Frame index')
				plt.ylabel('Window index')
				# plt.xticks(range(0, signal.shape[1], 5))
				pp.savefig()
				fh.show()
				fh.closeFigure()
		pp.close()

	if Config.showCorrelation:
		def showCorrelation(x, y, sx, sy, normalization):
			fh.openFigure('Correlation between ' + sx + ' and ' + sy + ' at layer ' + str(0) + ' - Normalization: ' + str(normalization), 1, (16, 9))
			# I = x.shape[0]
			c = np.zeros((I, x.shape[1]+y.shape[1]-1))
			for i in range(I):
				c[i] = correlate(x[i], y[i], normalization=normalization, removemean=True)
			cmax = np.max(np.abs(c))
			plt.imshow(c, extent=getExtent(x.shape[1], y.shape[1], I), cmap='bwr', vmin=-cmax, vmax=cmax, interpolation='none')
			plt.axis('auto')
			plt.xlabel('Time lag [frames]')
			plt.ylabel('Window index')
			plt.colorbar(label='Correlation')
			pp.savefig()
			fh.show()

		pp = PdfPages(fh.path + "5 Correlation.pdf")
		for m in range(len(data['sigsrc'])):
			for normalization in [None, 'unbiased', 'Pearson', 'Pearson-unbiased']:
				showCorrelation(data['displacement'], data['signal'][m, 0], 'displacement', trim(data['sigsrc'][m](0)), normalization)
				showCorrelation(dcum, data['signal'][m, 0], 'cumulative displacement', trim(data['sigsrc'][m](0)), normalization)
		for normalization in [None, 'unbiased', 'Pearson', 'Pearson-unbiased']:
			showCorrelation(data['signal'][0, 0], data['signal'][-1, 0], trim(data['sigsrc'][0](0)), trim(data['sigsrc'][-1](0)), normalization)
		pp.close()

path = 'FRET_sensors + actinHistamineExpt2'
# path = 'FRET_sensors + actinPDGFRhoA_multipoint_0.5fn_s3_good'
# path = 'GBD_sensors + actinExpt_01'
showSignals(path)