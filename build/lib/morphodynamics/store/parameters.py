from pathlib import Path
class Param:
    """Object storing relevant information regarding the processing,
    e.g. the window size, the analyzed signal, the type of segmentation used.
    """

    def __init__(
        self,
        data_folder=None,
        analysis_folder=None,
        seg_folder=None,
        T=100,
        data_type="series",
        n_curve=10000,
        morpho_name=None,
        signal_name=None,
        max_time=None,
        seg_algo="farid",
        lambda_=10,
        width=5,
        depth=5,
    ):

        """Standard __init__ method.
        Parameters
        ----------

        Attributes
        ----------
        """

        # Output directory
        if analysis_folder is not None:
            self.analysis_folder = Path(analysis_folder)
        else:
            self.analysis_folder = None

        # path to data
        if data_folder is not None:
            self.data_folder = Path(data_folder)
        else:
            self.data_folder = None

        # path to segmentation
        if seg_folder is not None:
            self.seg_folder = Path(seg_folder)
        else:
            self.seg_folder = None

        # set threshold
        self.T = T

        # type of data
        self.data_type = data_type

        # name of channel to use for segmentation
        self.morpho_name = morpho_name

        # names of signal channels
        self.signal_name = signal_name

        # Standard deviation for the Gaussian filter prior to segmentation; 0 deactivates filtering
        self.sigma = 2

        # Smoothing parameter for the spline curve representing the contour of the cell
        # self.lambda_ = 0
        self.lambda_ = lambda_
        # self.lambda_ = 1e3

        # Number of points in the spline
        self.n_curve = n_curve

        # # Number of sampling windows in the outer layer (along the curve)
        # # self.I = 48
        # self.I = 96
        #
        # # Number of sampling windows in the "radial" direction
        # self.J = 5

        # Dimensions of the sampling windows
        self.width = width
        self.depth = depth

        # max time
        self.max_time = max_time

        # should Z and T dimensions be switched
        #self.switch_TZ = switch_TZ

        # use segmentation from ilastik
        # self.ilastik = ilastik

        # what type of segmentation is used (currently: farid, cellpose or ilastik)
        self.seg_algo = seg_algo

        # cell location
        self.location = None

        # use cellpose
        # self.cellpose = False

        # cell diameter to use for segmentation
        self.diameter = 200

        # use distributed computing
        self.distributed = None

        # time step
        self.step = 1

        # bad frames
        self.bad_frames = []

        # path to random forest model
        self.random_forest = None

        # scalings to use with random forest
        self.scalings = [1,2]