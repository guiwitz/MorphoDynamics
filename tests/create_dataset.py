from pathlib import Path
from morphodynamics.morpho_segmentation import InteractSeg

if __name__ == '__main__':
    # create the dataset
    resultdir = Path('synthetic/data/Results_ilastik/')
    expdir = Path('synthetic/data')
    signal_name = ['synth_ch2.h5', 'synth_ch3.h5']
    morpho_name = 'synth_ch1.h5'
    self = InteractSeg(expdir=expdir, resultdir=resultdir, morpho_name=morpho_name, signal_name=signal_name)

    self.initialize()

    self.param.seg_algo = "ilastik"
    self.param.width = 5
    self.param.depth = 5
    self.param.lambda_ = 10

    self.run_segmentation()
    self.export_data()