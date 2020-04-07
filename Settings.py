import numpy as np
from skimage.external.tifffile import imread
from PIL import Image


class Struct:
    pass


class VirtualData:
    datadir = 'C:/Work/UniBE2/Data/'


class TIFFSeries(VirtualData):
    def __init__(self, name, expdir, morphofile, signalfile, K, shape):
        self.name = name
        self.expdir = expdir
        self.morphofile = morphofile
        self.signalfile = signalfile
        self.K = K
        self.shape = shape

    def load_frame_morpho(self, k):
        return imread(self.datadir + self.expdir + self.morphofile(k + 1) + '.tif').astype(dtype=np.uint16)

    def load_frame_signal(self, m, k):
        return imread(self.datadir + self.expdir + self.signalfile[m](k + 1) + '.tif').astype(dtype=np.uint16)

    def get_channel_name(self, m):
        return self.signalfile[m](0).split('/')[0]


class MultipageTIFF(VirtualData):
    def __init__(self, name, expdir, morphofile, signalfile, step):
        self.name = name
        self.expdir = expdir
        self.morphofile = morphofile
        self.signalfile = signalfile
        self.step = step

        self.morpho = Image.open(self.datadir + self.expdir + self.morphofile)
        self.K = self.morpho.n_frames // self.step
        self.shape = self.morpho.size

        self.signal = []
        for f in self.signalfile:
            self.signal.append(Image.open(self.datadir + self.expdir + f))

    def load_frame_morpho(self, k):
        self.morpho.seek(k*self.step)
        return np.array(self.morpho)

    def load_frame_signal(self, m, k):
        self.signal[m].seek(k*self.step)
        return np.array(self.signal[m])

    def get_channel_name(self, m):
        return self.signalfile[m].split('.')[0]


def load_settings(dataset_name):
    # Analysis parameters
    param = Struct()

    # Output directory
    param.resultdir = dataset_name + '/'

    # Standard deviation for the Gaussian filter prior to segmentation; 0 deactivates filtering
    param.sigma = 2
    if dataset_name == 'Ellipse with triangle dynamics':  # Skip Gaussian filtering for synthetic datasets
        param.sigma = 0

    # Smoothing parameter for the spline curve representing the contour of the cell
    # param.lambda_ = 0
    param.lambda_ = 1e2
    # param.lambda_ = 1e3

    # Number of sampling windows in the outer layer (along the curve)
    # param.I = 48
    param.I = 96

    # Number of sampling windows in the "radial" direction
    param.J = 5

    # Figure parameters
    param.showWindows = not True
    param.showCircularity = True
    param.showEdge = True
    param.showEdgePDF = True
    param.showDisplacement = True
    param.showSignals = True
    param.showCorrelation = True
    # param.edgeNormalization = 'global'
    param.edgeNormalization = 'frame-by-frame'

    if dataset_name == 'FRET_sensors + actinHistamineExpt2':
        expdir = 'FRET_sensors + actin/Histamine/Expt2/'
        morphofile = lambda k: 'w16TIRF-CFP/RhoA_OP_his_02_w16TIRF-CFP_t' + str(k)
        signalfile = [lambda k: 'ratio_tiffs/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(k, k, k),
                  lambda k: 'w16TIRF-CFP/RhoA_OP_his_02_w16TIRF-CFP_t' + str(k),
                  lambda k: 'w26TIRFFRETacceptor/RhoA_OP_his_02_w26TIRFFRETacceptor_t' + str(k),
                  lambda k: 'w26TIRFFRETacceptor_corr/RhoA_OP_his_02_w26TIRFFRETacceptor_t' + str(k),
                  lambda k: 'w34TIRF-mCherry/RhoA_OP_his_02_w34TIRF-mCherry_t' + str(k)]
        K = 159
        shape = (358, 358)
        dataset = TIFFSeries(dataset_name, expdir, morphofile, signalfile, K, shape)
        # param.T = None
        param.T = 222
    elif dataset_name == 'FRET_sensors + actinPDGFRhoA_multipoint_0.5fn_s3_good':
        expdir = 'FRET_sensors + actin/PDGF/RhoA_multipoint_0.5fn_s3_good/'
        morphofile = lambda k: 'w34TIRF-mCherry/RhoA_multipoint_0.5fn_01_w34TIRF-mCherry_s3_t' + str(k)
        signalfile = [lambda k: 'ratio_tiffs/photobleached_corrected_ratio_{:0>3d}'.format(k),
                  lambda k: 'w16TIRF-CFP/RhoA_multipoint_0.5fn_01_w16TIRF-CFP_s3_t' + str(k),
                  lambda k: 'w26TIRFFRETacceptor/RhoA_multipoint_0.5fn_01_w26TIRFFRETacceptor_s3_t' + str(k),
                  lambda k: 'w26TIRFFRETacceptor_corr/RhoA_multipoint_0.5fn_01_w26TIRFFRETacceptor_s3_t' + str(k),
                  lambda k: 'w34TIRF-mCherry/RhoA_multipoint_0.5fn_01_w34TIRF-mCherry_s3_t' + str(k)]
        K = 750
        shape = (358, 358)
        dataset = TIFFSeries(dataset_name, expdir, morphofile, signalfile, K, shape)
        param.T = 2620
    elif dataset_name == 'GBD_sensors + actinExpt_01':
        expdir = 'GBD_sensors + actin/Expt_01/'
        morphofile = lambda k: 'w14TIRF-GFP_s2/R52_LA-GFP_FN5_mCh-rGBD_02_w14TIRF-GFP_s2_t' + str(k)
        signalfile = [lambda k: 'w14TIRF-GFP_s2/R52_LA-GFP_FN5_mCh-rGBD_02_w14TIRF-GFP_s2_t' + str(k),
                  lambda k: 'w24TIRF-mCherry_s2/R52_LA-GFP_FN5_mCh-rGBD_02_w24TIRF-mCherry_s2_t' + str(k)]
        K = 250
        shape = (716, 716)
        dataset = TIFFSeries(dataset_name, expdir, morphofile, signalfile, K, shape)
        param.T = 165
    elif dataset_name == 'Ellipse with triangle dynamics':
        expdir = 'Synthetic data/Ellipse with triangle dynamics/'
        morphofile = lambda k: 'Morphology/Phantom' + str(k)
        signalfile = [lambda k: 'Signal/Phantom' + str(k)]
        K = 50
        shape = (101, 101)
        dataset = TIFFSeries(dataset_name, expdir, morphofile, signalfile, K, shape)
        param.T = None
    elif dataset_name == 'TIAM_protrusion':
        expdir = 'TIAM_protrusion/'
        morphofile = 'TIAM-nano_ACTIN.tif'
        signalfile = ['TIAM-nano_OPTO.tif', 'TIAM-nano_STIM.tif']
        step = 15
        dataset = MultipageTIFF(dataset_name, expdir, morphofile, signalfile, step)
        param.T = 150  # 126  # 140
        param.Tfun = lambda k: 150 * k / 65 + 162 * (65 - k) / 65
    elif dataset_name == 'TIAM_protrusion2':
        expdir = 'TIAM_protrusion/'
        morphofile = 'TIAM-nano_OPTO.tif'
        signalfile = ['TIAM-nano_OPTO.tif', 'TIAM-nano_STIM.tif']
        step = 15
        dataset = MultipageTIFF(dataset_name, expdir, morphofile, signalfile, step)
        param.T = 122  # 127 # 126  # 140
    # data.K = 3
    return dataset, param

# import matplotlib.pyplot as plt
# dataset_name = load_metadata('TIAM_protrusion')
# x = dataset_name.load_frame_morpho(0)
# plt.figure()
# plt.imshow(x, 'gray')
# plt.show()
