"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin) but your plugin may choose to implement any of the hook
specifications offered by napari.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below accordingly.  For complete documentation see:
https://napari.org/docs/dev/plugins/for_plugin_developers.html
"""
import numpy as np
from napari_plugin_engine import napari_hook_implementation
from ..store.dataset import MultipageTIFF

@napari_hook_implementation
def napari_get_reader(path):
    """A basic implementation of the napari_get_reader hook specification.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    #if not path.endswith(".tif"):
    #    return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """

    # load all files into array
    #image, channels = oirloader(path)

    data = MultipageTIFF(
                path,
                #morpho_name=self.param.morpho_name,
                #signal_name=self.param.signal_name,
                #data_type=self.param.data_type,
                #step=self.param.step,
                #bad_frames=self.param.bad_frames,
                #switch_TZ=self.param.switch_TZ,
                #max_time=self.param.max_time,
            )
    data.signal_name = data.channel_name
    image = data.load_frame_signal(0,0)
    
    #default_cols = ['magenta', 'cyan', 'yellow', 'gray'] + ['blue' for x in range(20)]
    #return [(image[:,:,i], {'name': channels[i], 'blending': 'additive', 'colormap': default_cols[i]}, 'image') for i in range(len(channels))]
    return image