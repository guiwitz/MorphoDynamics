import os

import numpy as np
import skimage
from PIL import Image
# from skimage.external.tifffile import TiffWriter, imread
from tifffile import TiffWriter, imread


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

    def load_frame_morpho(self, k):
        pass

    def load_frame_signal(self, m, k):
        pass

    def get_channel_name(self, m):
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
        print('Morpho: ' + str(self.morphofile(self.valid_frames[k] + 1)))
        return imread(self.datadir + self.expdir + self.morphofile(self.valid_frames[k] + 1) + '.tif').astype(dtype=np.uint16)

    def load_frame_signal(self, m, k):
        return imread(self.datadir + self.expdir + self.signalfile[m](self.valid_frames[k] + 1) + '.tif').astype(dtype=np.uint16)

    def get_channel_name(self, m):
        return self.signalfile[m](0).split('/')[0]


class TIFFSeriesAutodetect(VirtualData):
    def __init__(self, cur_dir, segm_folder, segment_files, signal_files):
        self.cur_dir = cur_dir
        self.segm_folder = segm_folder
        self.segment_files = segment_files
        self.signal_files = signal_files

    def load_frame_morpho(self, t):
        return skimage.io.imread(os.path.join(self.cur_dir, self.segm_folder, self.segment_files[t]))

    def load_frame_signal(self, channel_ind, t):
        return skimage.io.imread(os.path.join(self.cur_dir, channel_ind, self.signal_files[channel_ind][t]))

    def get_channel_name(self, m):
        # return self.signalfile[m](0).split('/')[0]
        return 'default'


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
        self.shape = (self.morpho.size[1], self.morpho.size[0])

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