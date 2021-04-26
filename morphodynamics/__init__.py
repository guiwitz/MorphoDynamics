#from .version import get_version
#__version__ = get_version()

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("morphodynamics").version
except DistributionNotFound:
    print('package is not installed')
    pass
