import numpy as np
from skimage.external.tifffile import imread, TiffWriter
from PIL import Image


class Struct:
    pass


class VirtualData:
    datadir = 'C:/Work/UniBE2/Data/'

    def set_frames(self, K, bad_frames):
        self.valid_frames = np.array(range(K))
        self.valid_frames[bad_frames] = -1
        self.valid_frames = self.valid_frames[self.valid_frames > -1]
        self.K = len(self.valid_frames)

    def dump_stack(self, m):
        tw = TiffWriter(self.get_channel_name(m) + '.tif')
        for k in range(self.K):
            x = self.load_frame_signal(m, k)
            tw.save(x, compress=6)
        tw.close()


class TIFFSeries(VirtualData):
    def __init__(self, name, expdir, morphofile, signalfile, shape, K, bad_frames=[]):
        self.name = name
        self.expdir = expdir
        self.morphofile = morphofile
        self.signalfile = signalfile
        self.shape = shape
        self.set_frames(K, bad_frames)

    def load_frame_morpho(self, k):
        return imread(self.datadir + self.expdir + self.morphofile(self.valid_frames[k] + 1) + '.tif').astype(dtype=np.uint16)

    def load_frame_signal(self, m, k):
        return imread(self.datadir + self.expdir + self.signalfile[m](self.valid_frames[k] + 1) + '.tif').astype(dtype=np.uint16)

    def get_channel_name(self, m):
        return self.signalfile[m](0).split('/')[0]

class MultipageTIFF(VirtualData):
    def __init__(self, name, expdir, morphofile, signalfile, step, bad_frames=[]):
        self.name = name
        self.expdir = expdir
        self.morphofile = morphofile
        self.signalfile = signalfile
        self.step = step

        self.morpho = Image.open(self.datadir + self.expdir + self.morphofile)
        self.set_frames(self.morpho.n_frames, bad_frames)
        self.K = self.K // self.step
        self.shape = self.morpho.size

        self.signal = []
        for f in self.signalfile:
            self.signal.append(Image.open(self.datadir + self.expdir + f))

    def load_frame_morpho(self, k):
        self.morpho.seek(self.valid_frames[k*self.step])
        return np.array(self.morpho)

    def load_frame_signal(self, m, k):
        self.signal[m].seek(self.valid_frames[k*self.step])
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
    param.showWindows = True
    param.showCircularity = True
    param.showEdge = True
    param.showEdgePDF = True
    param.showCurvature = True
    param.showDisplacement = True
    param.showSignals = True
    param.showCorrelation = True
    param.showFourierDescriptors = True
    # param.edgeNormalization = 'global'
    param.edgeNormalization = 'frame-by-frame'

    # param.showWindows = not True
    # param.showCircularity = not True
    # param.showEdge = not True
    # param.showEdgePDF = not True
    # param.showCurvature = not True
    # param.showDisplacement = True
    # param.showSignals = not True
    # param.showCorrelation = not True
    # param.showFourierDescriptors = not True
    # # param.edgeNormalization = 'global'
    # param.edgeNormalization = 'frame-by-frame'

    if dataset_name == 'FRET_sensors + actinHistamineExpt2':
        expdir = 'FRET_sensors + actin/Histamine/Expt2/'
        morphofile = lambda k: 'w16TIRF-CFP/RhoA_OP_his_02_w16TIRF-CFP_t' + str(k)
        signalfile = [lambda k: 'ratio_tiffs/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(k, k, k),
                  lambda k: 'w16TIRF-CFP/RhoA_OP_his_02_w16TIRF-CFP_t' + str(k),
                  lambda k: 'w26TIRFFRETacceptor/RhoA_OP_his_02_w26TIRFFRETacceptor_t' + str(k),
                  lambda k: 'w26TIRFFRETacceptor_corr/RhoA_OP_his_02_w26TIRFFRETacceptor_t' + str(k),
                  lambda k: 'w34TIRF-mCherry/RhoA_OP_his_02_w34TIRF-mCherry_t' + str(k)]
        shape = (358, 358)
        K = 159
        data = TIFFSeries(dataset_name, expdir, morphofile, signalfile, shape, K)
        # param.T = None
        param.T = 222
    elif dataset_name == 'FRET_sensors + actinHistamineExpt1_forPRES':
        expdir = 'FRET_sensors + actin/Histamine/Expt1_forPRES/'
        morphofile = lambda k: 'ratio_tiffs/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(k, k, k)
        signalfile = [
            lambda k: 'ratio_tiffs/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(k, k, k),
            lambda k: 'w34TIRF-mCherry/H1R_rhoa2g_01_w34TIRF-mCherry_s4_t' + str(k)]
        # morphofile = lambda k: 'ratio_tiffs/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(valid_frames[k-1+35]+1, valid_frames[k-1+35]+1, valid_frames[k-1+35]+1)
        # signalfile = [
        #     lambda k: 'ratio_tiffs/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(valid_frames[k-1+35]+1, valid_frames[k-1+35]+1, valid_frames[k-1+35]+1),
        #     lambda k: 'w34TIRF-mCherry/H1R_rhoa2g_01_w34TIRF-mCherry_s4_t' + str(valid_frames[k-1+35]+1)]
        shape = (358, 358)
        K = 125
        bad_frames = [41, 42, 50, 51, 52, 56, 57, 58]
        data = TIFFSeries(dataset_name, expdir, morphofile, signalfile, shape, K, bad_frames)
        # param.T = None
        # param.T = 1809
        # param.sigma = 0
        param.T = 1444.20
        param.sigma = 3
    elif dataset_name == 'FRET_sensors + actinPDGFRhoA_multipoint_0.5fn_s3_good':
        expdir = 'FRET_sensors + actin/PDGF/RhoA_multipoint_0.5fn_s3_good/'
        morphofile = lambda k: 'w34TIRF-mCherry/RhoA_multipoint_0.5fn_01_w34TIRF-mCherry_s3_t' + str(k)
        signalfile = [lambda k: 'ratio_tiffs/photobleached_corrected_ratio_{:0>3d}'.format(k),
                      lambda k: 'w16TIRF-CFP/RhoA_multipoint_0.5fn_01_w16TIRF-CFP_s3_t' + str(k),
                      lambda k: 'w26TIRFFRETacceptor/RhoA_multipoint_0.5fn_01_w26TIRFFRETacceptor_s3_t' + str(k),
                      lambda k: 'w26TIRFFRETacceptor_corr/RhoA_multipoint_0.5fn_01_w26TIRFFRETacceptor_s3_t' + str(k),
                      lambda k: 'w34TIRF-mCherry/RhoA_multipoint_0.5fn_01_w34TIRF-mCherry_s3_t' + str(k)]
        shape = (358, 358)
        K = 750
        data = TIFFSeries(dataset_name, expdir, morphofile, signalfile, shape, K)
        param.T = 2620
    elif dataset_name == 'FRET_sensors + actinPDGFExpt2_forPRES':
        expdir = 'FRET_sensors + actin/PDGF/Expt2_forPRES/'
        morphofile = lambda k: 'ratio_tiffs/photobleached_corrected_ratio_{:0>3d}'.format(k)
        signalfile = [lambda k: 'ratio_tiffs/photobleached_corrected_ratio_{:0>3d}'.format(k),
                      lambda k: 'w34TIRF-mCherry/RhoA_multipoint_0.5fn_01_w34TIRF-mCherry_s3_t' + str(k)]
        shape = (358, 358)
        K = 750
        bad_frames = [106, 111, 113]
        data = TIFFSeries(dataset_name, expdir, morphofile, signalfile, shape, K, bad_frames)
        # param.T = 3858  # For segmentation based on mCherry channel
        param.T = 1367.20  # For segmentation based on ratio_tiffs channel
        param.scaling_mean = [[1850, 2445], [1400, 21455]]
    elif dataset_name == 'GBD_sensors + actinExpt_01':
        expdir = 'GBD_sensors + actin/Expt_01/'
        morphofile = lambda k: 'w14TIRF-GFP_s2/R52_LA-GFP_FN5_mCh-rGBD_02_w14TIRF-GFP_s2_t' + str(k)
        signalfile = [lambda k: 'w14TIRF-GFP_s2/R52_LA-GFP_FN5_mCh-rGBD_02_w14TIRF-GFP_s2_t' + str(k),
                  lambda k: 'w24TIRF-mCherry_s2/R52_LA-GFP_FN5_mCh-rGBD_02_w24TIRF-mCherry_s2_t' + str(k)]
        shape = (716, 716)
        K = 250
        data = TIFFSeries(dataset_name, expdir, morphofile, signalfile, shape, K)
        param.T = 165
    elif dataset_name == 'Ellipse with triangle dynamics':
        expdir = 'Synthetic data/Ellipse with triangle dynamics/'
        morphofile = lambda k: 'Morphology/Phantom' + str(k)
        signalfile = [lambda k: 'Signal/Phantom' + str(k)]
        shape = (101, 101)
        K = 50
        data = TIFFSeries(dataset_name, expdir, morphofile, signalfile, shape, K)
        param.T = None
    elif dataset_name == 'TIAM_protrusion':
        expdir = 'CEDRIC_OPTOGENETICS/TIAM_protrusion/'
        morphofile = 'TIAM-nano_ACTIN.tif'
        signalfile = ['TIAM-nano_ACTIN.tif', 'TIAM-nano_OPTO.tif', 'TIAM-nano_STIM.tif']
        step = 15
        data = MultipageTIFF(dataset_name, expdir, morphofile, signalfile, step)
        param.T = 145  # 150
        param.Tfun = lambda k: 145 * k / 65 + 150 * (65 - k) / 65
        param.sigma = 5
    elif dataset_name == 'TIAM_protrusion_full':
        expdir = 'CEDRIC_OPTOGENETICS/TIAM_protrusion/'
        morphofile = 'TIAM-nano_ACTIN.tif'
        signalfile = ['TIAM-nano_ACTIN.tif', 'TIAM-nano_OPTO.tif', 'TIAM-nano_STIM.tif']
        step = 1
        bad_frames = [34, 50, 397, 446, 513, 684, 794, 962]
        data = MultipageTIFF(dataset_name, expdir, morphofile, signalfile, step, bad_frames)
        param.T = 145  # 150
        # param.Tfun = lambda k: 145 * k / 999 + 150 * (999 - k) / 999
        param.Tfun = lambda k: 145 * k / 999 + 155 * (999 - k) / 999
        param.sigma = 5
        param.scaling_disp = 15
    elif dataset_name == 'Rac1_arhgap31_01_s2_forPRES':
        expdir = 'CEDRIC_FA_FRET/Rac1_arhgap31_01/Rac1_arhgap31_01_s2_forPRES/'
        morphofile = lambda k: 'ratio_tiffs_nocorr/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(k, k, k)
        signalfile = [
            lambda k: 'ratio_tiffs_nocorr/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(k, k, k),
            lambda k: 'w34TIRF-mCherry_s2/Rac1_arhgap31_01_w34TIRF-mCherry_s2_t' + str(k)]
        shape = (716, 716)
        K = 300
        bad_frames = [25, 26]
        data = TIFFSeries(dataset_name, expdir, morphofile, signalfile, shape, K, bad_frames)
        # param.T = None
        param.T = 793  # 1232
    elif dataset_name == 'Rac1_arhgap31_02_s2_forPRES':
        expdir = 'CEDRIC_FA_FRET/Rac1_arhgap31_02/Rac1_arhgap31_02_s2_forPRES/'
        morphofile = lambda k: 'ratio_tiffs_nocor/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(k, k, k)
        signalfile = [
            lambda k: 'ratio_tiffs_nocor/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(k, k, k),
            lambda k: 'w34TIRF-mCherry_s2/Rac1_arhgap31_02_w34TIRF-mCherry_s2_t' + str(k)]
        K = 300
        shape = (716, 716)
        data = TIFFSeries(dataset_name, expdir, morphofile, signalfile, shape, K)
        # param.T = None
        param.T = 1038.90  # 1232
    # data.K = 10
    return data, param

# import matplotlib.pyplot as plt
# dataset_name = load_metadata('TIAM_protrusion')
# x = dataset_name.load_frame_morpho(0)
# plt.figure()
# plt.imshow(x, 'gray')
# plt.show()
