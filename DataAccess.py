import numpy as np
from PIL import Image
from skimage.external.tifffile import TiffWriter, imread


class VirtualData:
    datadir = 'C:/Work/UniBE2/Data/'

    def __init__(self, K, bad_frames):
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

    def get_channel_name(self, m):
        pass

    def load_frame_signal(self, m, k):
        pass


class TIFFSeries(VirtualData):
    def __init__(self, name, expdir, morphofile, signalfile, shape, K, bad_frames=[]):
        self.name = name
        self.expdir = expdir
        self.morphofile = morphofile
        self.signalfile = signalfile
        self.shape = shape
        super().__init__(K, bad_frames)

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
        super().__init__(self.morpho.n_frames, bad_frames)
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