import os
import pickle
from copy import deepcopy
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from pathlib import Path
import ipywidgets as ipw
from IPython.display import display, HTML, clear_output
import dill
from nd2reader import ND2Reader
import yaml

from .parameters import Param
from .dataset import MultipageTIFF, TIFFSeries, ND2, H5
from .folders import Folders

from .analysis_par import analyze_morphodynamics
from .windowing import (
    calculate_windows_index,
    create_windows,
    boundaries_image,
)
from .displacementestimation import rasterize_curve, splevper
from . import utils

from dask_jobqueue import SLURMCluster
from dask.distributed import Client, LocalCluster

import matplotlib
cmap2 = matplotlib.colors.ListedColormap (np.array([[1,0,0],[1,0,0]]))

# fix MacOSX OMP bug (see e.g. https://github.com/dmlc/xgboost/issues/1715)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# suppress figure titles in widgets rendering and enlarge notebook
display(HTML("<style>div.jupyter-widgets.widget-label {display: none;}</style>"))
display(HTML("<style>.container { width:100% !important; }</style>"))

style = {"description_width": "initial"}
layout = {"width": "300px"}

class InteractSeg:
    """
    Methods creating the user interface for the package.

    Parameters
    ----------
    data_folder: str
        path to folder containing data
    analysis_folder: str
        path to folder where to save data
    morpho_name: str
        name of folder or file used segmentation
    signal_name: list of str
        names of folders or files used as signal
    memory : str
        RAM to use on cluster
    cores : int
        number of cores to use per worker
    skip_trackseg : bool
        skip segmentation and tracking (only possible
        if done previously)
    seg_algo : str
        type of segmentation algorithm to use
        can be "fardi", "cellpose" or "ilastik"
    createUI : bool
        create or not a UI

    Attributes
    ----------
    data_folder, memory, cores: see Parameters
    """

    def __init__(
        self,
        data_folder=None,
        analysis_folder=None,
        seg_folder=None,
        seg_channel_name=None,
        signal_channel_names=None,
        memory="2 GB",
        cores=1,
        skip_trackseg=False,
        seg_algo="ilastik",
        do_createUI=True,
    ):

        directories = {
            'analysis_folder': analysis_folder,
            'data_folder': data_folder,
            'seg_folder': seg_folder}

        for d1, d2 in directories.items():
            if d2 is not None:
                directories[d1] = Path(d2)

        self.param = Param(
            data_folder=directories['data_folder'],
            analysis_folder=directories['analysis_folder'],
            seg_folder=directories['seg_folder'],
            morpho_name=seg_channel_name,
            signal_name=signal_channel_names,
            seg_algo=seg_algo
        )

        self.memory = memory
        self.cores = cores
        self.do_createUI = do_createUI
        self.skip_trackseg = skip_trackseg

        self.data = None
        self.res = None

        # create interactive browsers for data, analysis and segmentation folders
        self.data_folder_ipw = Folders(window_width=300, init_path=data_folder)
        self.analysis_folder_ipw = Folders(window_width=300, init_path=analysis_folder)
        self.seg_folder_ipw = Folders(window_width=300, init_path=seg_folder)

        # link interactive browsers to param file
        self.data_folder_ipw.file_list.observe(self.update_param_data_folder, names="value")
        self.analysis_folder_ipw.file_list.observe(self.update_param_analysis_folder, names="value")
        self.seg_folder_ipw.file_list.observe(self.update_param_seg_folder, names="value")

        # creat interactive list for channels
        self.channel_list_seg_ipw = ipw.Select(options=[])
        self.channel_list_signal_ipw = ipw.SelectMultiple(options=[])

        # update choices of segmentation and analysis folders when changin
        # main folder
        self.data_folder_ipw.file_list.observe(self.get_folders, names=("options", "value"))
        self.channel_list_seg_ipw.observe(self.update_param_morpho_name, names="value")
        self.channel_list_signal_ipw.observe(self.update_param_signal_name, names="value")
        self.get_folders() #force update

        # type of work: segmentation or loading
        self.switch_new_load = ipw.RadioButtons(
            value="New analysis",
            options=["New analysis", "Load analysis"],
            description="Usage type:"
        )
        self.switch_new_load.observe(self.ui, names="value")

        self.switch_reuse_seg = ipw.RadioButtons(
            value="New segmentation",
            options=["New segmentation", "Re-use segmentation"],
            description="Segmentation computation:"
        )
        self.switch_reuse_seg.observe(self.ui, names="value")

        # export, import buttons
        self.export_button = ipw.Button(description="Save analysis")
        self.export_button.on_click(self.export_data)

        self.load_button = ipw.Button(description="Load analysis")
        self.load_button.on_click(self.load_data)
        self.load_button.style.button_color = 'lightgreen'

        self.load_params_button = ipw.Button(description="Load parameters")
        self.load_params_button.on_click(self.load_params)

        # segmentation type
        self.seg_algo_ipw = ipw.RadioButtons(
            value=seg_algo,
            options=["farid", "cellpose", "ilastik"],
            description="Segmentation:",
        )
        self.seg_algo_ipw.observe(self.update_param_seg_algo, names="value")

        # create ipw outputs
        self.out_debug = ipw.Output()
        self.out = ipw.Output()
        self.out_distributed = ipw.Output()
        self.interface = ipw.Output()

        self.create_ui_options()

        if self.do_createUI:
            with self.out:

                self.fig, self.ax = plt.subplots(figsize=(5, 5))
                self.ax.set_title(f"Time:")

                self.implot = self.ax.imshow(np.zeros((2, 2)), cmap='gray')
                self.wplot = None#self.ax.imshow(np.zeros((20,20)), cmap=cmap2)
                self.tplt = None

                self.fig.show()

            self.ui()

        # some param values are rest when creating interactive features
        # we put them back to original values here
        if seg_channel_name is not None:
            self.param.morpho_name = seg_channel_name
            self.seg_folder_ipw.value = seg_channel_name
        if signal_channel_names is not None:
            self.param.signal_name = signal_channel_names
            self.channel_list_signal_ipw.value = signal_channel_names

    def create_ui_options(self):

        # slider for time limits to analyze
        self.time_slider = ipw.IntSlider(
            description="Time", min=0, max=0, value=0, continuous_update=True
        )
        self.time_slider.observe(self.show_segmentation, names="value")

        # intensity sliders
        self.intensity_range_slider = ipw.IntRangeSlider(
            description="Intensity range",
            min=0,
            max=0,
            value=(0, 0),
            continuous_update=False,
        )
        self.intensity_range_slider.observe(self.update_intensity_range, names="value")

        # channel to display
        self.display_channel_ipw = ipw.Select(options=[])
        self.channel_list_seg_ipw.observe(self.update_display_channel_list, names="value")
        self.channel_list_signal_ipw.observe(self.update_display_channel_list, names="value")
        self.display_channel_ipw.observe(self.update_display_channel, names="value")

        # show windows or nots
        self.show_windows_choice = ipw.Checkbox(description="Show windows", value=True)
        self.show_windows_choice.observe(self.update_windows_vis, names="value")

        # show windows or not
        self.show_text_choice = ipw.Checkbox(description="Show labels", value=True)
        self.show_text_choice.observe(self.update_text_vis, names="value")

        self.width_text = ipw.IntText(
            value=10, description="Window depth", layout=layout, style=style
        )
        self.width_text.observe(self.update_param_simple, names="value")

        self.depth_text = ipw.IntText(
            value=10, description="Window width", layout=layout, style=style
        )
        self.depth_text.observe(self.update_param_simple, names="value")

        # use distributed computing
        self.client = None
        self.distributed = ipw.Select(options=["local", "cluster"], value=None)
        self.distributed.observe(self.initialize_dask, names="value")

        # run the analysis button
        self.run_button = ipw.Button(description="Click to segment")
        self.run_button.on_click(self.run_segmentation)

        # load or initialize
        self.init_button = ipw.Button(description="Initialize data")
        self.init_button.on_click(self.initialize)
        self.init_button.style.button_color = 'lightgreen'

        # parameters
        self.max_time_ipw = ipw.BoundedIntText(value=0, description="Max time")
        self.max_time_ipw.observe(self.update_param_max_time, names="value")

        # parameters
        self.step_ipw = ipw.IntText(value=1, description="Step")
        self.step_ipw.observe(self.update_param_simple, names="value")

        # parameters
        self.bad_frames_ipw = ipw.Text(value="", description="Bad frames (e.g. 1,2,5-8,12)")
        self.bad_frames_ipw.observe(self.update_param_bad_frames, names="value")

        self.threshold_ipw = ipw.FloatText(value=100, description="Threshold:")
        self.threshold_ipw.observe(self.update_param_simple, names="value")

        self.diameter_ipw = ipw.FloatText(value=100, description="Diameter:")
        self.diameter_ipw.observe(self.update_param_simple, names="value")

        self.segparam_ipw = ipw.VBox([])

        self.location_x_ipw = ipw.FloatText(value=100, description="Location X")
        self.location_x_ipw.observe(self.update_location, names="value")

        self.location_y_ipw = ipw.FloatText(value=100, description="Location Y")
        self.location_y_ipw.observe(self.update_location, names="value")

    def initialize(self, b=None):
        """Create a data object based on chosen directories/files"""

        if os.path.isdir(os.path.join(self.param.data_folder, self.param.morpho_name)):
            self.param.data_type = "series"
            self.data = TIFFSeries(
                self.param.data_folder,
                self.param.morpho_name,
                self.param.signal_name,
                data_type=self.param.data_type,
                step=self.param.step,
                bad_frames=self.param.bad_frames,
                max_time=self.param.max_time,
            )
        elif self.param.morpho_name.split(".")[-1] == "tif":
            self.param.data_type = "multi"
            self.data = MultipageTIFF(
                self.param.data_folder,
                self.param.morpho_name,
                self.param.signal_name,
                data_type=self.param.data_type,
                step=self.param.step,
                bad_frames=self.param.bad_frames,
                switch_TZ=self.param.switch_TZ,
                max_time=self.param.max_time,
            )
        elif self.param.morpho_name.split(".")[-1] == "nd2":
            self.param.data_type = "nd2"
            self.data = ND2(
                self.param.data_folder,
                self.param.morpho_name,
                self.param.signal_name,
                data_type=self.param.data_type,
                step=self.param.step,
                bad_frames=self.param.bad_frames,
                max_time=self.param.max_time,
            )
        elif self.param.morpho_name.split(".")[-1] == "h5":
            self.param.data_type = "h5"
            self.data = H5(
                self.param.data_folder,
                self.param.morpho_name,
                self.param.signal_name,
                data_type=self.param.data_type,
                step=self.param.step,
                bad_frames=self.param.bad_frames,
                max_time=self.param.max_time,
            )

        self.max_time_ipw.max = self.data.max_time
        self.max_time_ipw.min = 0
        self.param.max_time = self.data.max_time
        self.max_time_ipw.value = self.data.max_time

        # display image
        if self.do_createUI:
            self.show_segmentation()

    def run_segmentation(self, b=None):
        """Run segmentation analysis"""

        if self.client is None:
            self.initialize_dask()
        self.run_button.description = "Segmenting..."
        # with self.out_debug:
        self.res = analyze_morphodynamics(
            self.data,
            self.param,
            self.client,
            skip_segtrack=self.skip_trackseg,
        )
        if self.do_createUI:
            self.show_segmentation()
        self.run_button.description = "Click to segment"

    def initialize_dask(self, change=None):
        """If dask is used, start it and display UI."""

        # with no manual selection use local. if param.distributed
        # has been set before, use that setting
        if self.distributed.value is None:
            self.distributed.value = "local"
        if self.param.distributed is None:
            self.param.distributed = self.distributed.value

        if self.param.distributed == "cluster":
            cluster = SLURMCluster(cores=self.cores, memory=self.memory)
            self.client = Client(cluster)
            if self.do_createUI:
                with self.out_distributed:
                    display(self.client.cluster._widget())
        elif self.param.distributed == "local":
            cluster = LocalCluster()
            # if self.cores is not None:
            #    cluster.scale(self.cores)
            self.client = Client(cluster)
            if self.do_createUI:
                with self.out_distributed:
                    display(self.client.cluster._widget())

    def windows_for_plot(self, N, im_shape, time):
        """Create a window image"""

        """c = rasterize_curve(
            N, im_shape, self.res.spline[time], self.res.orig[time]
        )"""
        save_path = os.path.join(self.param.analysis_folder, "segmented")
        c = skimage.io.imread(
            os.path.join(save_path, "rasterized_k_" + str(time) + ".tif")
        )

        w, _, _ = create_windows(
            c,
            splevper(self.res.orig[time], self.res.spline[time]),
            self.res.J,
            self.res.I,
        )
        return w

    def show_segmentation(self, change=None):
        """Update segmentation plot"""

        t = self.time_slider.value
        # load image of channel selected in self.display_channel_ipw
        image = self.load_image(t)

        # calculate windows
        b0 = None
        windows_pos = None
        if self.res is not None:
            # window = self.windows_for_plot(self.param.n_curve, image.shape, t)
            name = os.path.join(
                self.param.analysis_folder,
                "segmented",
                "window_k_" + str(t) + ".pkl",
            )
            window = pickle.load(open(name, "rb"))

            name = os.path.join(
                self.param.analysis_folder,
                "segmented",
                "window_image_k_" + str(t) + ".tif",
            )
            # b0 = boundaries_image(image.shape, window)
            b0 = skimage.io.imread(name)
            b0 = b0.astype(float)
            b0[b0 == 0] = np.nan
            windows_pos = np.array(calculate_windows_index(window))

        # self.fig.axes[0].set_title(f"Time:{self.data.valid_frames[t]}")
        with self.out:
            self.implot.set_array(image)

            # update max slider value and current value
            self.intensity_range_slider.unobserve_all()
            self.intensity_range_slider.max = np.max(
                [int(image.max()), self.intensity_range_slider.max])
            if self.intensity_range_slider.value[1] == 0:
                self.intensity_range_slider.value = (0, int(image.max()))
            self.intensity_range_slider.observe(
                self.update_intensity_range, names="value"
            )
            self.implot.set_clim(vmin=self.intensity_range_slider.value[0], vmax=self.intensity_range_slider.value[1])

            #adjust plot size if created
            if self.implot.get_extent()[1] < 3:
                self.implot.set_extent((0, image.shape[1], 0, image.shape[0]))
            if b0 is not None:
                if self.wplot is None:
                    self.wplot = self.ax.imshow(b0, cmap=cmap2)
                    self.wplot.set_extent((0, image.shape[1], 0, image.shape[0]))
                else:
                    self.wplot.set_array(b0)

            if windows_pos is not None:
                if self.tplt is None:
                    self.tplt = [self.ax.text(
                        p[0],
                        p[1],
                        int(p[2]),
                        color="green",
                        fontsize=10,
                        horizontalalignment="center",
                        verticalalignment="center",
                    ) for p in windows_pos]
                else:
                    for ind, p in enumerate(windows_pos):
                        self.tplt[ind].set_x(p[0])
                        self.tplt[ind].set_y(image.shape[0]-p[1])
                        self.tplt[ind].set_text(str(int(p[2])))

    def load_image(self, time):
        """Load image selected in self.display_channel_ipw widget"""

        if self.display_channel_ipw.value == self.param.morpho_name:
            image = self.data.load_frame_morpho(time)
        else:
            channel_index = self.param.signal_name.index(self.display_channel_ipw.value)
            image = self.data.load_frame_signal(channel_index, time)

        return image

    def get_folders(self, change=None):
        """Update channel options when selecting new main folder"""

        # if nd2 file, load and use metadata channels as choices
        if len(self.data_folder_ipw.file_list.value) > 0:
            if (self.data_folder_ipw.file_list.value[0]).split(".")[-1] == "nd2":
                image = ND2Reader(
                    os.path.join(
                        self.data_folder_ipw.cur_dir,
                        self.data_folder_ipw.file_list.value[0],
                    )
                )
                self.channel_list_seg_ipw.options = image.metadata["channels"]
                self.channel_list_seg_ipw.value = None

                self.channel_list_signal_ipw.options = image.metadata["channels"]
                self.channel_list_signal_ipw.value = ()

                self.param.data_folder = self.data_folder_ipw.cur_dir.joinpath(
                    self.data_folder_ipw.file_list.value[0],
                )
        else:
            folder_names = np.array(
                [x for x in self.data_folder_ipw.file_list.options if x[0] != "."]
            )

            # self.channel_list_seg_ipw.unobserve(self.update_file_list, names='value')
            self.channel_list_seg_ipw.options = folder_names
            self.channel_list_seg_ipw.value = None
            # self.channel_list_seg_ipw.observe(self.update_file_list, names='value')

            self.channel_list_signal_ipw.options = folder_names
            self.param.data_folder = self.data_folder_ipw.cur_dir

    def update_display_channel_list(self, change=None):
        """Callback to update available channels to display"""

        self.display_channel_ipw.options = [str(self.channel_list_seg_ipw.value)] + [
            str(x) for x in self.channel_list_signal_ipw.value
        ]
        self.display_channel_ipw.value = str(self.channel_list_seg_ipw.value)

    def update_display_channel(self, change=None):
        """Callback to update displayed channel"""
        if self.data is not None:
            self.show_segmentation()

    def update_param_simple(self, change=None):
        """Calback to update segmentation file lists depending on selections"""

        self.param.step = self.step_ipw.value
        self.param.threshold = self.threshold_ipw.value
        self.param.diameter = self.diameter_ipw.value
        self.param.width = self.width_text.value
        self.param.depth = self.depth_text.value
        self.param.T = self.threshold_ipw.value

    def update_param_max_time(self, change=None):

        self.param.max_time = self.max_time_ipw.value
        self.data.update_params(self.param)
        self.time_slider.max = self.data.K-1

    def update_param_bad_frames(self, change=None):
        # parse bad frames
        bads = utils.format_bad_frames(self.bad_frames_ipw.value)
        self.param.bad_frames = bads

        # if data object does not exist, create it now
        if self.data is None:
            self.initialize()

        # update params
        self.data.update_params(self.param)
        self.time_slider.max = self.data.K - 1

    def update_param_morpho_name(self, change=None):
        """Calback to update segmentation file lists depending on selections"""

        self.param.morpho_name = str(self.channel_list_seg_ipw.value)

    def update_param_signal_name(self, change=None):
        """Calback to update signal file lists depending on selections"""

        self.param.signal_name = [str(x) for x in self.channel_list_signal_ipw.value]

    def update_param_data_folder(self, change=None):
        """Callback to update saving directory paramters"""

        self.param.data_folder = self.data_folder_ipw.cur_dir

    def update_param_analysis_folder(self, change=None):
        """Callback to update saving directory paramters"""

        self.param.analysis_folder = self.analysis_folder_ipw.cur_dir

    def update_param_seg_folder(self, change=None):
        """Callback to update saving directory paramters"""

        self.param.seg_folder = self.seg_folder_ipw.cur_dir

    def update_param_seg_algo(self, change=None):
        """Callback to update saving directory paramters"""

        self.param.seg_algo = self.seg_algo_ipw.value
        if self.seg_algo_ipw.value == "cellpose":
            self.segparam_ipw = self.diameter_ipw
        elif self.seg_algo_ipw.value == "farid":
            self.segparam_ipw = ipw.VBox([])
        elif self.seg_algo_ipw.value == "ilastik":
            self.segparam = ipw.VBox([])
        self.ui()

    def update_intensity_range(self, change=None):
        """Callback to update intensity range"""

        self.intensity_range_slider.max = self.implot.get_array().max()
        self.implot.set_clim(vmin=self.intensity_range_slider.value[0], vmax=self.intensity_range_slider.value[1])

    def update_location(self, change=None):
        """Callback to update cm location after manual location entering"""

        self.param.location = [self.location_x_ipw.value, self.location_y_ipw.value]
        self.show_segmentation()

    def update_windows_vis(self, change=None):
        """Callback to turn windows visibility on/off"""
        if self.wplot.get_alpha() == 0:
            self.wplot.set_alpha(1)
        else:
            self.wplot.set_alpha(0)

    def update_text_vis(self, change=None):
        """Callback to turn windows labels visibility on/off"""

        if self.tplt[0].get_alpha() == 0:
            turnonoff = [t.set_alpha(1) for t in self.tplt]
        else:
            turnonoff = [t.set_alpha(0) for t in self.tplt]

    def on_key_press(self, event):
        """Record if shift key is pressed"""

        if event.key == "shift":
            self.shift_is_held = True

    def on_key_release(self, event):
        """Record if shift key is released"""
        if event.key == "shift":
            self.shift_is_held = False

    def export_data(self, b=None):
        """Callback to export Results and Parameters"""

        if self.data.data_type == "nd2":
            del self.data.nd2file

        dill.dump(
            self.res,
            open(os.path.join(self.param.analysis_folder, "Results.pkl"), "wb"),
        )

        dict_file = {}
        for x in dir(self.param):
            if x[0] == "_":
                None
            elif (x == "analysis_folder") or (x == "data_folder") or (x == "seg_folder"):
                dict_file[x] = getattr(self.param, x).as_posix()
            else:
                dict_file[x] = getattr(self.param, x)

        dict_file["bad_frames"] = self.bad_frames_ipw.value

        with open(self.param.analysis_folder.joinpath("Parameters.yml"), "w") as file:
            yaml.dump(dict_file, file)

        print("Your results have been saved in the following directory:")
        print(self.param.analysis_folder)

    def load_data(self, b=None):
        """Callback to load params, data and results"""

        folder_load = self.analysis_folder_ipw.cur_dir

        self.param, self.res, self.data = utils.load_alldata(
            folder_load, load_results=True
        )
        self.param.bad_frames = utils.format_bad_frames(self.param.bad_frames)

        param_copy = deepcopy(self.param)

        if self.do_createUI:
            self.update_interface(param_copy)
            self.show_segmentation()

    def load_params(self, b=None):
        """Callback to load only params and data """

        folder_load = self.analysis_folder_ipw.cur_dir
        self.param, _, self.data = utils.load_alldata(folder_load, load_results=False)

        param_copy = deepcopy(self.param)
        self.update_interface(param_copy)

    def update_interface(self, param_copy):
        """Set interface parameters using information from param object"""

        # set paths, folders and channel selectors
        if self.data is not None:
            if self.data.data_type == "nd2":
                self.data_folder_ipw.go_to_folder(param_copy.data_folder.parent)
                self.channel_list_seg_ipw.options = [param_copy.morpho_name]
                self.channel_list_signal_ipw.options = param_copy.signal_name
            else:
                self.data_folder_ipw.go_to_folder(param_copy.data_folder)

        self.channel_list_seg_ipw.value = param_copy.morpho_name
        self.channel_list_signal_ipw.value = param_copy.signal_name

        # set segmentation type
        self.seg_algo_ipw.value = param_copy.seg_algo

        # set segmentation parameters
        self.threshold_ipw.value = param_copy.T
        self.diameter_ipw.value = param_copy.diameter
        if param_copy.location is not None:
            self.location_x_ipw.value = param_copy.location[0]
            self.location_y_ipw.value = param_copy.location[1]
        self.width_text.value = param_copy.width
        self.depth_text.value = param_copy.depth
        self.max_time_ipw.value = param_copy.max_time
        self.bad_frames_ipw.value = param_copy.bad_frames_txt

        if self.data is not None:
            self.time_slider.max = self.data.K - 1
        self.step_ipw.value = param_copy.step

        self.update_display_channel_list()

    def ui(self, change=None):
        """Create interface"""

        if self.switch_new_load.value == "New analysis":
            self.folder_panel = self.ui_folder_panel('new')
        else:
            self.folder_panel = self.ui_folder_panel('load')

        with self.interface:
            clear_output()
            display(
                ipw.VBox(
                    [
                        ipw.HTML(
                            '<br><font size="5"><b>1. Are you running a new analysis or loading one?<b></font>'
                        ),
                        self.switch_new_load,
                        self.folder_panel,
                        ipw.HTML('<br><font size="5"><b>Run or visualize segmentation<b></font>'),
                        ipw.HBox(
                            [
                                ipw.VBox(
                                    [
                                        self.max_time_ipw,
                                        self.step_ipw,
                                        self.bad_frames_ipw,
                                        self.seg_algo_ipw,
                                        self.segparam_ipw,
                                        self.location_x_ipw,
                                        self.location_y_ipw,
                                        self.width_text,
                                        self.depth_text,
                                        self.run_button,
                                    ]
                                ),
                                self.out,
                                #self.fig,  #
                                ipw.VBox(
                                    [
                                        self.time_slider,
                                        self.intensity_range_slider,
                                        self.show_windows_choice,
                                        self.show_text_choice,
                                        self.display_channel_ipw,
                                    ]
                                ),
                            ]
                        ),
                        self.export_button,
                    ]
                )
            )

    def ui_folder_panel(self, work_type):

        if work_type == "new":

            sel_data = ipw.VBox([
                ipw.HTML('<font size="3"><b>(a) Data folder<b></font>'),
                self.data_folder_ipw.file_list])

            sel_results = ipw.VBox([
                ipw.HTML('<font size="3"><b>(b) Results folder<b></font>'),
                self.analysis_folder_ipw.file_list])

            sel_segmentation = ipw.VBox([
                ipw.HTML('<font size="3"><b>(c) Segmentation folder<b></font>'),
                self.seg_folder_ipw.file_list])

            sel_channels = ipw.HBox([
                ipw.VBox([
                    ipw.HTML('<font size="3"><b>(a) Segmentation<b></font>'),
                    self.channel_list_seg_ipw
                ]),
                ipw.VBox([
                    ipw.HTML('<font size="3"><b>(b) Signal<b></font>'),
                    self.channel_list_signal_ipw
                ]),
            ])

            panel = ipw.VBox(
                [
                    ipw.HTML('<br><font size="5"><b>2. Select locations.<b></font>\
                        <br><font size="3"><b>Select the folders (a) containing your data, (b) where your results \
                    should be saved, and (c) where the segmentation either is already available or should be saved.<b></font>'),
                    ipw.HBox([sel_data, sel_results, sel_segmentation]),
                    ipw.HTML('<br><font size="5"><b>3. Choose channels<b></font>\
                        <br><font size="3"><b>Select the channel that you want to use for (a) Segmentation \
                            and the channel(s) from which you want to extract (b) Signal.\
                                <br>Even if you use a pre-computed segmentation, specify a segmentation channel.\
                                    <p style="color:lightgreen;">Once you are done, hit the Initialize button.</p><b></font>'),
                    sel_channels,
                    self.init_button,
                    ipw.HTML('<br><font size="5"><b>4. Computing type<b></font>\
                            <br><font size="3"><b>If you run the software on a local computer, you can specify\
                                how many cores you want to use. If you run on a SLURM cluster, you can \
                                    specifiy how many parallel jobs you want to use.<b></font>'),
                    self.distributed,
                    self.out_distributed,
                ])

        else:
            panel = ipw.VBox([
                ipw.HTML('<br><font size="5"><b>2. Select locations<b></font>\
                    <br><font size="3"><b>Choose the location of the analysis that you want to import. That folder should\
                        contain e.g. a Parameters.yml file.\
                            <p style="color:lightgreen;">Once you are done, hit the "Load analysis" button.</p>. .<b></font>'),
                self.analysis_folder_ipw.file_list,
                self.load_button
                ])

        return panel
