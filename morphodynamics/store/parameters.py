from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, List

@dataclass
class Param:
    """Object storing relevant information regarding the processing,
    e.g. the window size, the analyzed signal, the type of segmentation used.

    Parameters
    ----------
    data_folder: location of data
    analysis_folder: location of analysis
    seg_folder: location of segmentation
    data_type: type of data, currently 'series', 'multi', 'nd2', 'h5' or 'np'
    n_curve: number of points in the spline
    morpho_name: name of channel to use for segmentation
    signal_name: names of signal channels
    max_time: maximum time to analyze
    step: time step
    seg_algo: segmentation algorithm to use, currently 'farid', 'cellpose', 'ilastik' or 'conv_paint'
    threshold: threshold for segmentation
    sigma: standard deviation for the Gaussian filter prior to segmentation; 0 deactivates filtering
    location: location of the cell to analyze in pixels (x,y)
    diameter: diameter of the cell to analyze in pixels, used for cellpose segmentation
    lambda_: smoothing parameter for the spline curve representing the contour of the cell
    random_forest: path to random forest model, usd for conv_paint segmentation
    width: width of the sampling windows
    depth: depth of the sampling windows

    """

    data_folder: Union[Path, str] = None
    analysis_folder: Union[Path, str] = None
    seg_folder: Union[Path, str] = None
    data_type: str = "series"
    n_curve: int = 10000
    morpho_name: str = None
    signal_name: List[str] = None
    max_time: int = None
    bad_frames: List[int] = field(default_factory=list)
    step: int = 1
    seg_algo: str = "cellpose"
    threshold: int = None
    sigma: int = 2
    location: List[int] = None
    diameter: int = 200
    random_forest: Union[Path, str] = None
    lambda_: float = 10
    width: int = 5
    depth: int = 5


    def __post_init__(self):

        # Output directory
        if self.analysis_folder is not None:
            self.analysis_folder = Path(self.analysis_folder)

        # path to data
        if self.data_folder is not None:
            self.data_folder = Path(self.data_folder)

        # path to segmentation
        if self.seg_folder is not None:
            self.seg_folder = Path(self.seg_folder)