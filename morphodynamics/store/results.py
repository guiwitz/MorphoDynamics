import numpy as np


class Results:
    """
    Object storing relevant information regarding the processing,
    e.g. the window size, the analyzed signal, the type of segmentation used.

    Parameters
    ----------
    J: int
        number of window layers
    I: list of int
        number of windows in each layer
    num_time_points: int
        number of analyzed time points
    num_channels: int
        number of analyzed channels
    location: list of int
        x,y location of initial cell


    Attributes
    ----------
    seg: list of 2d arrays
        segmentation masks
    spline: list of spline objects
        splines for each time point
    spline_param0: list of arrays
        arrays of spline-parameters defining points for
        frame=t matching points at frame=t+1 defined by spline_param
    spline_param: list of arrays
        arrays of spline-parameters defining points for
        frame=t-1 matching points at frame=t defined by spline_param0
    displacement: 2d array
        contour displacement between consecutive frames. Dimensions
        number of windows x number of time points
    mean: 4d array
        extracted mean signal for each layer(L) window (W), time point(T),
        and signal (S). Dimensions are C x L x W x T
    var: 4d array
        extracted variance signal for each layer(L) window (W), time point(T),
        and signal (S). Dimensions are C x L x W x T
    length: 1d array
        cell contour length for each frame
    area: 1d array
        cell area for each frame
    orig; 1d array
        shift of spline parameter to align spline origin on
        origin of previous spline

    """

    def __init__(self, J, I, num_time_points, num_channels, location=None):

        # Number of window layers
        self.J = J

        # Number of windows per layer
        self.I = I
        Imax = np.max(I)

        #
        self.location = location

        # segmentation masks
        self.seg = []

        # List of spline objects for each time point
        self.spline = []

        # spline_param0 and spline_param are lists of spline parameters
        # defining matching pairs between consecutive time points. E.g.
        # spline_param0[0] and spline_param[0] correspond to points
        # at t=0 and t=1.
        self.spline_param0 = []
        self.spline_param = []

        # projected displacements
        self.displacement = np.zeros((I[0], num_time_points - 1))

        # extracted mean values from windows in all channels
        self.mean = np.zeros((num_channels, J, Imax, num_time_points))

        # extracted variance values from windows in all channels
        self.var = np.zeros((num_channels, J, Imax, num_time_points))

        # cell contour length
        self.length = np.zeros((num_time_points,))

        # cell area
        self.area = np.zeros((num_time_points,))

        # spline origin shift
        self.orig = np.zeros((num_time_points,))
