import numpy as np

from microfilm.dataset import TIFFSeries as TIFFSeries_or
from microfilm.dataset import MultipageTIFF as MultipageTIFF_or
from microfilm.dataset import ND2 as ND2_or
from microfilm.dataset import H5 as H5_or
from microfilm.dataset import Nparray as Nparray_or
from types import MethodType


def load_frame_morpho(self, k):
    """Load index k of valid frames of the segmentation channel"""

    if self.morpho_name is not None:
        time = self.valid_frames[k]
        image = self.load_frame(self.morpho_name, time)
        return image.astype(dtype=np.uint16)
    else:
        raise Exception(f"Sorry, no segmentation channel has been provided.")
        
def load_frame_signal(self, m, k):
    """Load index k of valid frames of channel index m in self.signal_name"""

    if self.signal_name is not None:
        time = self.valid_frames[k]
        image = self.load_frame(self.signal_name[m], time)
        return image.astype(dtype=np.uint16)
    else:
        raise Exception(f"Sorry, no signal channel has been provided.")


class TIFFSeries(TIFFSeries_or):
    def __init__(
        self,
        expdir,
        channel_name=None,
        bad_frames=[],
        step=1,
        max_time=None,
        data_type=None,
        morpho_name=None,
        signal_name=None,
    ):
        super().__init__(
            expdir,
            channel_name=channel_name,
            bad_frames=bad_frames,
            step=step,
            max_time=max_time,
            data_type=data_type,
        )
        self.morpho_name = morpho_name
        self.signal_name = signal_name
        self.load_frame_morpho = MethodType(load_frame_morpho, self)
        self.load_frame_signal = MethodType(load_frame_signal, self)
        self.num_timepoints = self.K

        
class MultipageTIFF(MultipageTIFF_or):
    def __init__(
        self,
        expdir,
        channel_name=None,
        bad_frames=[],
        step=1,
        max_time=None,
        data_type=None,
        morpho_name=None,
        signal_name=None,
    ):
        super().__init__(
            expdir,
            channel_name=channel_name,
            bad_frames=bad_frames,
            step=step,
            max_time=max_time,
            data_type=data_type,
        )
        self.morpho_name = morpho_name
        self.signal_name = signal_name
        self.load_frame_morpho = MethodType(load_frame_morpho, self)
        self.load_frame_signal = MethodType(load_frame_signal, self)
        self.num_timepoints = self.K

class ND2(ND2_or):
    def __init__(
        self,
        expdir,
        channel_name=None,
        bad_frames=[],
        step=1,
        max_time=None,
        data_type=None,
        morpho_name=None,
        signal_name=None,
    ):
        super().__init__(
            expdir,
            channel_name=channel_name,
            bad_frames=bad_frames,
            step=step,
            max_time=max_time,
            data_type=data_type,
        )
        self.morpho_name = morpho_name
        self.signal_name = signal_name
        self.load_frame_morpho = MethodType(load_frame_morpho, self)
        self.load_frame_signal = MethodType(load_frame_signal, self)
        self.num_timepoints = self.K
        
class H5(H5_or):
    def __init__(
        self,
        expdir,
        channel_name=None,
        bad_frames=[],
        step=1,
        max_time=None,
        data_type=None,
        morpho_name=None,
        signal_name=None,
    ):
        super().__init__(
            expdir,
            channel_name=channel_name,
            bad_frames=bad_frames,
            step=step,
            max_time=max_time,
            data_type=data_type,
        )
        self.morpho_name = morpho_name
        self.signal_name = signal_name
        self.load_frame_morpho = MethodType(load_frame_morpho, self)
        self.load_frame_signal = MethodType(load_frame_signal, self)
        self.num_timepoints = self.K

class Nparray(Nparray_or):
    def __init__(
        self,
        nparray,
        expdir,
        channel_name=None,
        bad_frames=[],
        step=1,
        max_time=None,
        data_type=None,
        morpho_name=None,
        signal_name=None,
    ):
        super().__init__(
            nparray,
            expdir,
            channel_name=channel_name,
            bad_frames=bad_frames,
            step=step,
            max_time=max_time,
            data_type=data_type,
        )
        self.morpho_name = morpho_name
        self.signal_name = signal_name
        #self.load_frame_morpho = MethodType(load_frame_morpho, self)
        self.load_frame_signal = MethodType(load_frame_signal, self)
        self.num_timepoints = self.K

    def load_frame_morpho(self, k):
        """Load index k of valid frames of the segmentation channel"""
        
        if self.morpho_name is not None:
                time = self.valid_frames[k]
                image = self.load_frame(self.morpho_name, time)
                image = image.astype(np.float32)
                image = (image - image.min()) / (image.max() - image.min())
                return image
        else:
            raise Exception(f"Sorry, no segmentation channel has been provided.")