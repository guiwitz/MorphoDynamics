from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("morphodynamics").version
except DistributionNotFound:
    # package is not installed
    pass

from .napari_plugin.napari_gui import MorphoWidget
from napari_plugin_engine import napari_hook_implementation

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [MorphoWidget]