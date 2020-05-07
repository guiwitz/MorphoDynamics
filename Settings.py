from DataAccess import TIFFSeries, MultipageTIFF


class Struct:
    pass


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
    param.showWindows = False
    param.showCircularity = False
    param.showEdgeOverview = False
    param.showEdgeVectorial = False
    param.showEdgeRasterized = False
    param.showCurvature = False
    param.showDisplacement = False
    param.showSignals = False
    param.showCorrelation = False
    param.showFourierDescriptors = False
    # param.edgeNormalization = 'global'
    param.edgeNormalization = 'frame-by-frame'

    # param.showWindows = True
    # param.showCircularity = True
    # param.showEdgeOverview = True
    # param.showEdgeVectorial = True
    param.showEdgeRasterized = True
    # param.showCurvature = True
    # param.showDisplacement = True
    # param.showSignals = True
    # param.showCorrelation = True
    # param.showFourierDescriptors = True

    if dataset_name == 'Ellipse with triangle dynamics':
        expdir = 'Synthetic data/Ellipse with triangle dynamics/'
        morphofile = lambda k: 'Morphology/Phantom' + str(k)
        signalfile = [lambda k: 'Signal/Phantom' + str(k)]
        shape = (101, 101)
        K = 50
        data = TIFFSeries(dataset_name, expdir, morphofile, signalfile, shape, K)
        param.T = None
    elif dataset_name == 'Change of origin':
        expdir = 'Synthetic data/Change of origin/'
        morphofile = lambda k: 'Phantom' + str(k)
        signalfile = []
        shape = (101, 101)
        K = 50
        data = TIFFSeries(dataset_name, expdir, morphofile, signalfile, shape, K)
        param.T = None
        param.I = 12
        param.J = 3
    elif dataset_name == 'FRET_sensors + actinHistamineExpt2':
        expdir = 'FRET_sensors + actin/Histamine/Expt2/'
        morphofile = lambda k: 'w16TIRF-CFP/RhoA_OP_his_02_w16TIRF-CFP_t' + str(k)
        signalfile = [
            lambda k: 'ratio_tiffs/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(k, k, k),
            lambda k: 'w16TIRF-CFP/RhoA_OP_his_02_w16TIRF-CFP_t' + str(k),
            lambda k: 'w26TIRFFRETacceptor/RhoA_OP_his_02_w26TIRFFRETacceptor_t' + str(k),
            lambda k: 'w26TIRFFRETacceptor_corr/RhoA_OP_his_02_w26TIRFFRETacceptor_t' + str(k),
            lambda k: 'w34TIRF-mCherry/RhoA_OP_his_02_w34TIRF-mCherry_t' + str(k)]
        shape = (358, 358)
        K = 159
        data = TIFFSeries(dataset_name, expdir, morphofile, signalfile, shape, K)
        # param.T = None
        param.T = 222
    elif dataset_name == 'FRET_sensors + actinMigration_line_1D':
        expdir = 'FRET_sensors + actin/Migration_line_1D/'
        morphofile = lambda k: 'ratio_tiffs/photobleached_corrected_ratio_{:0>4d}'.format(k)
        signalfile = [
            lambda k: 'ratio_tiffs/photobleached_corrected_ratio_{:0>4d}'.format(k),
            lambda k: 'w34TIRF-mCherry/RhoA_on_line_chemo_w34TIRF-mCherry_t' + str(k)]
        shape = (358, 358)
        K = 550  # 600
        data = TIFFSeries(dataset_name, expdir, morphofile, signalfile, shape, K)
        param.T = 1299
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
    elif dataset_name == 'GBD_sensors + actinExpt_01':
        expdir = 'GBD_sensors + actin/Expt_01/'
        morphofile = lambda k: 'w14TIRF-GFP_s2/R52_LA-GFP_FN5_mCh-rGBD_02_w14TIRF-GFP_s2_t' + str(k)
        signalfile = [lambda k: 'w14TIRF-GFP_s2/R52_LA-GFP_FN5_mCh-rGBD_02_w14TIRF-GFP_s2_t' + str(k),
                  lambda k: 'w24TIRF-mCherry_s2/R52_LA-GFP_FN5_mCh-rGBD_02_w24TIRF-mCherry_s2_t' + str(k)]
        shape = (716, 716)
        K = 250
        data = TIFFSeries(dataset_name, expdir, morphofile, signalfile, shape, K)
        param.T = 165
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
        # param.Tfun = lambda k: 145 * k / 999 + 155 * (999 - k) / 999
        param.Tfun = lambda k: 145 * k / 991 + 155 * (991 - k) / 991
        param.sigma = 5
        param.scaling_disp = 15
    elif dataset_name == 'FRET_sensors + actinHistamineExpt1_forPRES':
        expdir = 'FRET_sensors + actin/Histamine/Expt1_forPRES/'
        morphofile = lambda k: 'ratio_tiffs/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(k, k, k)
        signalfile = [
            lambda k: 'ratio_tiffs/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(k, k, k),
            lambda k: 'w34TIRF-mCherry/H1R_rhoa2g_01_w34TIRF-mCherry_s4_t' + str(k)]
        shape = (358, 358)
        K = 125
        bad_frames = [41, 42, 50, 51, 52, 56, 57, 58]
        data = TIFFSeries(dataset_name, expdir, morphofile, signalfile, shape, K, bad_frames)
        # param.T = None
        # param.T = 1809
        # param.sigma = 0
        param.T = 1444.20
        param.sigma = 3
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
    elif dataset_name == 'Rac1_arhgap31_01_s2_forPRES':
        expdir = 'CEDRIC_FA_FRET/Rac1_arhgap31_01/Rac1_arhgap31_01_s2_forPRES/'
        morphofile = lambda k: 'ratio_tiffs_nocorr/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(k, k, k)
        signalfile = [
            lambda k: 'ratio_tiffs/photobleached_corrected_ratio_{:0>3d}'.format(k),
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
            lambda k: 'ratio_tiffs/photobleached_corrected_ratio_{:0>3d}'.format(k),
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
