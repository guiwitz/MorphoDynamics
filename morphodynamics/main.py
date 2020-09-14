# Run this file if you want to execute the code as a script (without the graphical user interface).

import numpy as np
import os
import dill
from .analysis import analyze_morphodynamics
from .artifactgeneration import show_analysis
from .settings import load_settings

# Specify the dataset to be analyzed
# dataset_name = 'Ellipse with triangle dynamics'
# dataset_name = 'Change of origin'
# dataset_name = 'Moving ellipse'
dataset_name = 'FRET_sensors + actinHistamineExpt2'
# dataset_name = 'FRET_sensors + actinHistamineExpt2 with Cellpose'
# dataset_name = 'FRET_sensors + actinMigration_line_1D'
# dataset_name = 'FRET_sensors + actinPDGFRhoA_multipoint_0.5fn_s3_good'
# dataset_name = 'GBD_sensors + actinExpt_01'
# dataset_name = 'TIAM_protrusion'
# dataset_name = 'TIAM_protrusion with Cellpose'
# dataset_name = 'TIAM_protrusion_full'
# dataset_name = 'FRET_sensors + actinHistamineExpt1_forPRES'
# dataset_name = 'FRET_sensors + actinPDGFExpt2_forPRES'
# dataset_name = 'Rac1_arhgap31_01_s2_forPRES'
# dataset_name = 'Rac1_arhgap31_02_s2_forPRES'
# dataset_name = 'H1R_rGBD_01_forpres'
# dataset_name = 'H1R_rGBD_03_forpres'
# dataset_name = 'H1R_rGBD_03_forpres_end'

# Load the dataset and parameters defined in Settings.py
data, param = load_settings(dataset_name)

if not os.path.exists(param.resultdir):
    os.mkdir(param.resultdir)

# 1 = analyze data; 2 = generate figures; 0 = do both
step = 0

if step in [0, 1]:
    dill.dump(param, open(param.resultdir + 'Parameters.pkl', 'wb'))
    res = analyze_morphodynamics(data, param)
    dill.dump(res, open(param.resultdir + 'Results.pkl', 'wb'))

if step in [0, 2]:
    res = dill.load(open(param.resultdir + "Results.pkl", "rb"))
    show_analysis(data, param, res)
