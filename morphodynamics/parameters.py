class Param:
    """Object storing relevant information regarding the processing,
    e.g. the window size, the analyzed signal, the type of segmentation used.
    """

    def __init__(
        self,
        expdir=None,
        resultdir=None,
        T=None,
        data_type="series",
        n_curve=10000,
        morpho_name=None,
        signal_name=None,
        max_time=None,
    ):

        """Standard __init__ method.
        Parameters
        ----------

        Attributes
        ----------
        """

        # Output directory
        self.resultdir = resultdir

        # type of data
        self.data_type = data_type

        # path to data
        self.expdir = expdir

        # set threshold
        self.T = T

        # name of channel to use for segmentation
        self.morpho_name = morpho_name

        # names of signal channels
        self.signal_name = signal_name

        # Standard deviation for the Gaussian filter prior to segmentation; 0 deactivates filtering
        self.sigma = 2

        # Smoothing parameter for the spline curve representing the contour of the cell
        # self.lambda_ = 0
        self.lambda_ = 1e2
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
        self.width = 10
        self.depth = 10

        # max time
        self.max_time = max_time

        # cell location
        self.location = None

        # use cellpose
        self.cellpose = False

        # use distributed computing
        self.distributed = False

        # time step
        self.step = 1

        # bad frames
        self.bad_frames = []

        # Figure parameters
        self.showSegmentation = False
        self.showWindows = False
        self.showCircularity = False
        self.showEdgeOverview = False
        self.showEdgeVectorial = False
        self.showEdgeRasterized = False
        self.showCurvature = False
        self.showDisplacement = False
        self.showSignals = False
        self.showCorrelation = False
        self.showFourierDescriptors = False
        self.edgeNormalization = "frame-by-frame"
