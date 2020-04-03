import os
import dill
from Analysis import analyze_morphodynamics
from ArtifactGeneration import show_analysis
from Metadata import Struct, load_data
from Segmentation import segment

# Dataset specification
# dataset_name = 'Ellipse with triangle dynamics'
# dataset_name = 'FRET_sensors + actinHistamineExpt2'
# dataset_name = 'FRET_sensors + actinPDGFRhoA_multipoint_0.5fn_s3_good'
# dataset_name = 'GBD_sensors + actinExpt_01'
dataset_name = 'TIAM_protrusion'

# Analysis parameters
I = 48  # Number of sampling windows in the outer layer (along the curve)
J = 5  # Number of sampling windows in the "radial" direction
smooth_image = dataset_name != 'Ellipse with triangle dynamics'  # Gaussian smoothing (not for synthetic datasets)
show_win = False # Graphical representation of the windows

# Figure parameters
config = Struct()
config.showCircularity = True
config.showEdge = True
config.showEdgePDF = True
config.showDisplacement = True
config.showSignals = True
config.showCorrelation = True
# config.edgeNormalization = 'global'
config.edgeNormalization = 'frame-by-frame'

data = load_data(dataset_name)
# data.K = 3

# x = data.load_frame_morpho(500)
# for T in range(120, 160):
#     segment(x, T, smooth_image=True)
# quit()

res = analyze_morphodynamics(data, I, J, smooth_image, show_win)

if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)
dill.dump(res, open(dataset_name + '/Results.pkl', 'wb'))  # Save analysis results to disk

res = dill.load(open(dataset_name + "/Results.pkl", "rb"))

show_analysis(data, config, res)