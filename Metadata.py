class Struct:
    pass

def load_metadata(dataset):
    metadata = Struct()
    if dataset == 'FRET_sensors + actinHistamineExpt2':
        metadata.expdir = 'FRET_sensors + actin/Histamine/Expt2/'
        metadata.morphodir = lambda k: 'w16TIRF-CFP/RhoA_OP_his_02_w16TIRF-CFP_t' + str(k)
        metadata.sigdir = [lambda k: 'ratio_tiffs/ratio_bs_shade_corrected_{:0>3d}_to_bs_shade_corrected_{:0>3d}_{:0>3d}'.format(k, k, k),
                  lambda k: 'w16TIRF-CFP/RhoA_OP_his_02_w16TIRF-CFP_t' + str(k),
                  lambda k: 'w26TIRFFRETacceptor/RhoA_OP_his_02_w26TIRFFRETacceptor_t' + str(k),
                  lambda k: 'w26TIRFFRETacceptor_corr/RhoA_OP_his_02_w26TIRFFRETacceptor_t' + str(k),
                  lambda k: 'w34TIRF-mCherry/RhoA_OP_his_02_w34TIRF-mCherry_t' + str(k)]
        metadata.K = 159
        metadata.shape = (358, 358)
        # metadata.T = None
        metadata.T = 222
    elif dataset == 'FRET_sensors + actinPDGFRhoA_multipoint_0.5fn_s3_good':
        metadata.expdir = 'FRET_sensors + actin/PDGF/RhoA_multipoint_0.5fn_s3_good/'
        metadata.morphodir = lambda k: 'w34TIRF-mCherry/RhoA_multipoint_0.5fn_01_w34TIRF-mCherry_s3_t' + str(k)
        metadata.sigdir = [lambda k: 'ratio_tiffs/photobleached_corrected_ratio_{:0>3d}'.format(k),
                  lambda k: 'w16TIRF-CFP/RhoA_multipoint_0.5fn_01_w16TIRF-CFP_s3_t' + str(k),
                  lambda k: 'w26TIRFFRETacceptor/RhoA_multipoint_0.5fn_01_w26TIRFFRETacceptor_s3_t' + str(k),
                  lambda k: 'w26TIRFFRETacceptor_corr/RhoA_multipoint_0.5fn_01_w26TIRFFRETacceptor_s3_t' + str(k),
                  lambda k: 'w34TIRF-mCherry/RhoA_multipoint_0.5fn_01_w34TIRF-mCherry_s3_t' + str(k)]
        metadata.K = 750
        metadata.shape = (358, 358)
        metadata.T = 2620
    elif dataset == 'GBD_sensors + actinExpt_01':
        metadata.expdir = 'GBD_sensors + actin/Expt_01/'
        metadata.morphodir = lambda k: 'w14TIRF-GFP_s2/R52_LA-GFP_FN5_mCh-rGBD_02_w14TIRF-GFP_s2_t' + str(k)
        metadata.sigdir = [lambda k: 'w14TIRF-GFP_s2/R52_LA-GFP_FN5_mCh-rGBD_02_w14TIRF-GFP_s2_t' + str(k),
                  lambda k: 'w24TIRF-mCherry_s2/R52_LA-GFP_FN5_mCh-rGBD_02_w24TIRF-mCherry_s2_t' + str(k)]
        metadata.K = 250
        metadata.shape = (716, 716)
        metadata.T = 165
    elif dataset == 'Ellipse with triangle dynamics':
        metadata.expdir = 'Synthetic data/Ellipse with triangle dynamics/'
        metadata.morphodir = lambda k: 'Morphology/Phantom' + str(k)
        metadata.sigdir = [lambda k: 'Signal/Phantom' + str(k)]
        metadata.K = 50
        metadata.shape = (101, 101)
        metadata.T = None
    return metadata
