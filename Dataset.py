import os
import re
from aicsimageio import AICSImage
from nd2reader import ND2Reader
import skimage.io
import numpy as np
from pathlib import Path


class Data:
    def __init__(self, expdir, morpho_name, signal_name, bad_frames=[], step=1, max_time=None,
                 data_type='series', signalfile=None, morphofile=None):

        """Standard __init__ method.
        Parameters
        ----------
        expdir: str
            path to data folder (tif) or file (ND2)
        morpho_name: str
            name of data to use for segmentation
        signal_name: list of str
            names of data to use as signals
        bad_frames: list
            list of time-points to discard
        step: int
            time step to use when iterating across frames
        max_time: int
            last frame to consider
        data_type: str
            type of data considers ("series", "multi", "nd2")
        morphofile: str or list
            'series': 
                list of str, each element is a filename for a specific frame
            'multi':
                str, filename for a specific channel
            'nd2':
                str, channel name matching metadata information
        signalfile: str or list
            'series': 
                list of list of str, each element is a filename,
                files are grouped in a list for each channel, and 
                all list grouped in a larger list
            'multi':
                list of str, each element is a filename corresponding
                to a channnel
            'nd2':
                list of str, each element is a channel name matching
                metadata information
        
        Attributes
        ----------
        K: int
            Number of frames considerd (including steps, bad frames)
        dims: tuple
            XY image dimensions
        valid_frames: 1D array
            indices of conisered frames
        """

        self.data_type = data_type
        self.expdir = Path(expdir)
        # folder name (series) or file name (multipage)
        self.morpho_name = morpho_name
        # list of folder names (series) or list of file names (multipage)
        self.signal_name = signal_name
        self.bad_frames = np.array(bad_frames)
        self.step = step
        self.signalfile = signalfile
        self.morphofile = morphofile
        self.max_time = max_time
        self.K = None
        self.dims = None
        
    def set_valid_frames(self):
        self.valid_frames = np.arange(self.max_time)
        self.valid_frames = self.valid_frames[~np.in1d(self.valid_frames, self.bad_frames)]
        self.valid_frames = self.valid_frames[::self.step]
        self.K = len(self.valid_frames)
        
    def find_files(self, folderpath):
        '''Given a folder, parse contents to find all time points'''

        image_names = os.listdir(folderpath)
        image_names = np.array([x for x in image_names if x[0] != '.'])
        if len(image_names) > 0:
            times = [re.findall('.*\_t*(\d+)\.tif', x) for x in image_names]
            times = [int(x[0]) for x in times if len(x) > 0]
            image_names = image_names[np.argsort(times)]
        return image_names
    
    def update_params(self, params):
        
        self.max_time = params.max_time
        self.step = params.step
        self.bad_frames = params.bad_frames
        self.set_valid_frames()
        
        
class TIFFSeries(Data):
    def __init__(self, expdir, morpho_name, signal_name, bad_frames=[], step=1,
                 max_time=None, data_type='series', signalfile=None, morphofile=None):
        Data.__init__(self, expdir, morpho_name, signal_name, bad_frames, step,
                 max_time, data_type, signalfile, morphofile)

        self.initialize()
        
    def initialize(self):

        self.morphofile = self.find_files(os.path.join(self.expdir, self.morpho_name))
        self.signalfile = [self.find_files(os.path.join(self.expdir, x)) for x in self.signal_name]

        if self.max_time is None:
            self.max_time = len(self.morphofile)
        
        self.set_valid_frames()

        image = self.load_frame_morpho(0)
        self.dims = image.shape
        self.shape = image.shape
        
    def load_frame_morpho(self, k):
        
        time = self.valid_frames[k] + 1
        full_path = os.path.join(self.expdir, self.morpho_name, self.morphofile[time])
        return skimage.io.imread(full_path).astype(dtype=np.uint16)
    
    def load_frame_signal(self, m, k):

        time = self.valid_frames[k] + 1
        full_path = os.path.join(self.expdir, self.signal_name[m], self.signalfile[m][time])
        return skimage.io.imread(full_path).astype(dtype=np.uint16)
    
    def get_channel_name(self, m):
        return self.signalfile[m][0].split('.')[0]

    
class MultipageTIFF(Data):
    def __init__(self, expdir, morpho_name, signal_name, bad_frames=[], step=1,
                 max_time=None, data_type='multi', signalfile=None, morphofile=None):
        Data.__init__(self, expdir, morpho_name, signal_name, bad_frames, step,
                 max_time, data_type, signalfile, morphofile)

        self.initialize()
        
    def initialize(self):
        self.morphofile = self.morpho_name
        self.signalfile = self.signal_name

        self.morpho_imobj = AICSImage(os.path.join(self.expdir, self.morphofile))
        self.signal_imobj = [AICSImage(os.path.join(self.expdir, x)) for x in self.signal_name]

        if self.max_time is None:
            self.max_time = self.morpho_imobj.size_z
        
        self.set_valid_frames()

        image = self.load_frame_morpho(0)
        self.dims = image.shape
        self.shape = image.shape
        
    def load_frame_morpho(self, k):
        
        time = self.valid_frames[k] + 1
        image = self.morpho_imobj.get_image_data('YX', S=0, T=0, C=0, Z=time)
        return image.astype(dtype=np.uint16)
    
    def load_frame_signal(self, m, k):

        time = self.valid_frames[k] + 1
        
        image = self.signal_imobj[m].get_image_data('YX', S=0, T=0, C=0, Z=time)
        return image.astype(dtype=np.uint16)
    
    def get_channel_name(self, m):
        return self.signalfile[m].split('.')[0]

    
class ND2(Data):
    def __init__(self, expdir, morpho_name, signal_name, bad_frames=[], step=1,
                 max_time=None, data_type='nd2', signalfile=None, morphofile=None):
        Data.__init__(self, expdir, morpho_name, signal_name, bad_frames, step,
                 max_time, data_type, signalfile, morphofile)

        self.initialize()

    def initialize(self):
                        
        self.morphofile = self.morpho_name
        self.signalfile = self.signal_name
        
        #self.expdir = Path(self.expdir)

        self.nd2file = ND2Reader(self.expdir)
        self.nd2file.metadata['z_levels'] = range(0)
        
        if self.max_time is None:
            self.max_time = self.nd2file.sizes['t']

        self.set_valid_frames()

        image = self.load_frame_morpho(0)
        self.dims = image.shape
        self.shape = image.shape

    def load_frame_morpho(self, k):
        
        time = self.valid_frames[k] + 1
        
        ch_index = self.nd2file.metadata['channels'].index(self.morphofile)
        image = self.nd2file.get_frame_2D(x=0, y=0, z=0, c=ch_index, t=time, v=0)
        return image
        

    def load_frame_signal(self, m, k):

        time = self.valid_frames[k] + 1

        ch_index = self.nd2file.metadata['channels'].index(self.signalfile[m])
        image = self.nd2file.get_frame_2D(x=0, y=0, z=0, c=ch_index, t=time, v=0)
        return image
    

    def get_channel_name(self, m):
        return self.signalfile[m]


    