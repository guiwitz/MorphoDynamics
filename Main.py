import os
import dill
from Analysis import analyze_morphodynamics
from ArtifactGeneration import show_analysis
from Settings import Struct, load_settings
from Segmentation import segment

# dataset_name = 'Ellipse with triangle dynamics'
# dataset_name = 'FRET_sensors + actinHistamineExpt2'
# dataset_name = 'FRET_sensors + actinPDGFRhoA_multipoint_0.5fn_s3_good'
# dataset_name = 'GBD_sensors + actinExpt_01'
dataset_name = 'TIAM_protrusion2'

data, param = load_settings(dataset_name)

# x = data.load_frame_morpho(500)
# for T in range(120, 160):
#     segment(x, T, smooth_image=True)
# quit()

if not os.path.exists(param.resultdir):
    os.mkdir(param.resultdir)

res = analyze_morphodynamics(data, param)

dill.dump(res, open(param.resultdir + 'Results.pkl', 'wb'))  # Save analysis results to disk

# quit()

res = dill.load(open(param.resultdir + "Results.pkl", "rb"))

show_analysis(data, param, res)