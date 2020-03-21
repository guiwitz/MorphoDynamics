def loadMetadata(dataset):
    if dataset == 'FRET_sensors + actinHistamineExpt2':
        path = 'C:/Work/UniBE2/Guillaume/Example_Data/FRET_sensors + actin/Histamine/Expt2/'
        morphosrc = 'w16TIRF-CFP/RhoA_OP_his_02_w16TIRF-CFP_t'
        sigsrc = [lambda k: 'ratio_tiffs/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(k, k, k),
                  lambda k: 'w16TIRF-CFP/RhoA_OP_his_02_w16TIRF-CFP_t' + str(k),
                  lambda k: 'w26TIRFFRETacceptor/RhoA_OP_his_02_w26TIRFFRETacceptor_t' + str(k),
                  lambda k: 'w26TIRFFRETacceptor_corr/RhoA_OP_his_02_w26TIRFFRETacceptor_t' + str(k),
                  lambda k: 'w34TIRF-mCherry/RhoA_OP_his_02_w34TIRF-mCherry_t' + str(k)]
        K = 159
        shape = (358, 358)
        T = None
    elif dataset == 'FRET_sensors + actinPDGFRhoA_multipoint_0.5fn_s3_good':
        path = 'C:/Work/UniBE2/Guillaume/Example_Data/FRET_sensors + actin/PDGF/RhoA_multipoint_0.5fn_s3_good/'
        morphosrc = 'w34TIRF-mCherry/RhoA_multipoint_0.5fn_01_w34TIRF-mCherry_s3_t'
        sigsrc = [lambda k: 'ratio_tiffs/photobleached_corrected_ratio_{:0>3d}'.format(k),
                  lambda k: 'w16TIRF-CFP/RhoA_multipoint_0.5fn_01_w16TIRF-CFP_s3_t' + str(k),
                  lambda k: 'w26TIRFFRETacceptor/RhoA_multipoint_0.5fn_01_w26TIRFFRETacceptor_s3_t' + str(k),
                  lambda k: 'w26TIRFFRETacceptor_corr/RhoA_multipoint_0.5fn_01_w26TIRFFRETacceptor_s3_t' + str(k),
                  lambda k: 'w34TIRF-mCherry/RhoA_multipoint_0.5fn_01_w34TIRF-mCherry_s3_t' + str(k)]
        K = 750
        shape = (358, 358)
        T = 2620
    elif dataset == 'GBD_sensors + actinExpt_01':
        path = 'C:/Work/UniBE2/Guillaume/Example_Data/GBD_sensors + actin/Expt_01/'
        morphosrc = 'w14TIRF-GFP_s2/R52_LA-GFP_FN5_mCh-rGBD_02_w14TIRF-GFP_s2_t'
        sigsrc = [lambda k: 'w14TIRF-GFP_s2/R52_LA-GFP_FN5_mCh-rGBD_02_w14TIRF-GFP_s2_t' + str(k),
                  lambda k: 'w24TIRF-mCherry_s2/R52_LA-GFP_FN5_mCh-rGBD_02_w24TIRF-mCherry_s2_t' + str(k)]
        K = 250
        shape = (716, 716)
        T = 165
    elif dataset == 'Phantom':
        path = 'C:/Work/UniBE2/Code/Output 1/'
        morphosrc = 'protrudingEllipse/Phantom'
        sigsrc = [lambda k: 'signalEllipse/Phantom' + str(k)]
        K = 50
        shape = (101, 101)
        T = None
    else:
        path = 'Walking rectangles/'
        morphosrc = 'Phantom'
        sigsrc = []
        K = 50
        shape = (101, 101)
        T = None
    return path, morphosrc, sigsrc, K, T