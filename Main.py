import os
import dill
from Analysis import analyze_morphodynamics
from ArtifactGeneration import show_analysis
from Metadata import Struct, load_metadata

# Dataset specification
dataset = 'Ellipse with triangle dynamics'
# dataset = 'FRET_sensors + actinHistamineExpt2'
# dataset = 'FRET_sensors + actinPDGFRhoA_multipoint_0.5fn_s3_good'
# dataset = 'GBD_sensors + actinExpt_01'

# Analysis parameters
I = 48  # Number of sampling windows in the outer layer (along the curve)
J = 5  # Number of sampling windows in the "radial" direction
smooth_image = dataset != 'Ellipse with triangle dynamics'  # Gaussian smoothing (not for synthetic datasets)
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

md = load_metadata(dataset)

res = analyze_morphodynamics(dataset, md, I, J, smooth_image, show_win)

if not os.path.exists(dataset):
    os.mkdir(dataset)
dill.dump(res, open(dataset + '/Data.pkl', 'wb'))  # Save analysis results to disk

res = dill.load(open(dataset + "/Data.pkl", "rb"))

show_analysis(dataset, md, config, res)