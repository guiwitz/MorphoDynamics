import os
import dill
from Analysis import analyze_morphodynamics
from ArtifactGeneration import show_analysis
from Settings import Struct, load_settings
from Segmentation import segment

# dataset_name = 'Ellipse with triangle dynamics'
# dataset_name = 'FRET_sensors + actinHistamineExpt1_forPRES'
# dataset_name = 'FRET_sensors + actinHistamineExpt2'
# dataset_name = 'FRET_sensors + actinPDGFRhoA_multipoint_0.5fn_s3_good'
# dataset_name = 'FRET_sensors + actinPDGFExpt2_forPRES'
# dataset_name = 'GBD_sensors + actinExpt_01'
# dataset_name = 'TIAM_protrusion'
dataset_name = 'TIAM_protrusion_full'
# dataset_name = 'Rac1_arhgap31_01_s2_forPRES'
# dataset_name = 'Rac1_arhgap31_02_s2_forPRES'

data, param = load_settings(dataset_name)

# x = data.load_frame_morpho(500)
# for T in range(120, 160):
#     segment(x, T, smooth_image=True)
# quit()

if not os.path.exists(param.resultdir):
    os.mkdir(param.resultdir)

# import numpy as np
# from skimage.filters import gaussian
# y = np.zeros((data.K,) + data.shape, dtype=np.float32)
# for k in range(data.K):
#     y[k] = gaussian(data.load_frame_morpho(k), sigma=data.sigma, preserve_range=True)

step = 2

if step in [0, 1]:
    res = analyze_morphodynamics(data, param)
    dill.dump(param, open(param.resultdir + 'Parameters.pkl', 'wb'))
    dill.dump(res, open(param.resultdir + 'Results.pkl', 'wb'))

if step in [0, 2]:
    res = dill.load(open(param.resultdir + "Results.pkl", "rb"))
    show_analysis(data, param, res)
