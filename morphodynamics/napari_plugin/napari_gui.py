"""
This module implements a napari widget to create microfilm images interactively
by capturing views.
"""

import pickle
from itertools import cycle
from pathlib import Path
from PyQt5.QtWidgets import QGridLayout
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import (QWidget, QPushButton, QSpinBox,
QVBoxLayout, QLabel, QComboBox, QCheckBox,
QTabWidget, QListWidget, QFileDialog, QScrollArea, QAbstractItemView)

import numpy as np
import napari

from dask import delayed
import dask.array as da
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, LocalCluster

from .folder_list_widget import FolderListWidget
from ..store.parameters import Param
from ..utils import dataset_from_param, load_alldata, export_results_parameters
from ..analysis_par import analyze_morphodynamics, segment_single_frame, compute_spline_windows
from ..windowing import label_windows
from napari_convpaint import ConvPaintWidget
from .VHGroup import VHGroup


class MorphoWidget(QWidget):
    """
    Implentation of a napari plugin offering an interface to the morphodynamics softwere.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The napari viewer object.
    """
    
    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer

        # create a param object
        self.param = Param(
            seg_algo='cellpose'
        )
        self.analysis_path = None
        self.cluster = None

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self.setMinimumWidth(400)
        self.tabs = QTabWidget()
        self._layout.addWidget(self.tabs)

        # main tab
        self.main = QWidget()
        self._main_layout = QVBoxLayout()
        self.main.setLayout(self._main_layout)
        self.tabs.addTab(self.main, 'Main')

        # options tab
        self.options = QWidget()
        self._options_layout = QVBoxLayout()
        self.options.setLayout(self._options_layout)
        self.tabs.addTab(self.options, 'Options')
        
        # conv paint tab
        self.paint = QWidget()
        self._paint_layout = QVBoxLayout()
        self.paint.setLayout(self._paint_layout)
        self.tabs.addTab(self.paint, 'Conv paint')

        # display tab
        self.display_options = QWidget()
        self._display_options_layout = QGridLayout()
        self.display_options.setLayout(self._display_options_layout)
        self.tabs.addTab(self.display_options, 'Display options')

        # dask tab
        self.dask = QWidget()
        self._dask_layout = QVBoxLayout()
        self.dask.setLayout(self._dask_layout)
        self.tabs.addTab(self.dask, 'Dask')

        # add convpaint widget
        #self.deep_paint_widget = DeepPaintWidget(self.viewer, self.param)
        self.conv_paint_widget = ConvPaintWidget(self.viewer)
        self._paint_layout.addWidget(self.conv_paint_widget)

        # add widgets to main tab
        self.data_vgroup = VHGroup('1. Select location of data', orientation='G')
        self._main_layout.addWidget(self.data_vgroup.gbox)

        # files
        self.file_list = FolderListWidget(napari_viewer)
        self.data_vgroup.glayout.addWidget(self.file_list)
        self.file_list.setMaximumHeight(100)

        # Pick folder to analyse interactively
        btn_select_file_folder = QPushButton("Select data folder")
        btn_select_file_folder.clicked.connect(self._on_click_select_file_folder)
        self.data_vgroup.glayout.addWidget(btn_select_file_folder)

        # channel selection
        self.segm_channel = QListWidget()
        self.segm_channel.setMaximumHeight(100)
        self.signal_channel = QListWidget()
        self.signal_channel.setMaximumHeight(100)
        self.signal_channel.setSelectionMode(QAbstractItemView.ExtendedSelection)

        channel_group = VHGroup('2. Select channels to use', orientation='G')
        self._main_layout.addWidget(channel_group.gbox)

        channel_group.glayout.addWidget(QLabel('Segmentation'),0,0)
        channel_group.glayout.addWidget(QLabel('Signal'),0,1)
        channel_group.glayout.addWidget(self.segm_channel,1,0)
        channel_group.glayout.addWidget(self.signal_channel,1,1)

        # load data
        load_group = VHGroup('3. Load and display the dataset', orientation='G')
        self._main_layout.addWidget(load_group.gbox)
        self.btn_load_data = QPushButton("Load")
        load_group.glayout.addWidget(self.btn_load_data)

        # select saving place
        analysis_vgroup = VHGroup('4. Set location to save analysis', 'G')
        analysis_vgroup.gbox.setMaximumHeight(100)
        self._main_layout.addWidget(analysis_vgroup.gbox)

        self.btn_select_analysis = QPushButton("Set analysis folder")
        self.display_analysis_folder = QLabel("No selection")
        self.display_analysis_folder.setWordWrap(True)
        #, self.scroll_analysis = scroll_label('No selection.')
        analysis_vgroup.glayout.addWidget(self.display_analysis_folder, 0, 0)
        analysis_vgroup.glayout.addWidget(self.btn_select_analysis, 0, 1)
        
        segmentation_group = VHGroup('Alternative segmentation', 'G')
        segmentation_group.gbox.setMaximumHeight(150)
        
        self._options_layout.addWidget(segmentation_group.gbox)
        self.btn_select_segmentation = QPushButton("Set segmentation folder")
        self.display_segmentation_folder, self.scroll_segmentation = scroll_label('No selection.')
        segmentation_group.glayout.addWidget(self.scroll_segmentation, 0, 0)
        segmentation_group.glayout.addWidget(self.btn_select_segmentation, 1, 0)

        # load analysis
        self.btn_load_analysis = QPushButton("Load analysis")
        self._options_layout.addWidget(self.btn_load_analysis)

        self.settings_vgroup = VHGroup('5. Set analysis settings and run', orientation='G')
        self._main_layout.addWidget(self.settings_vgroup.gbox)

        # algo choice
        self.seg_algo = QComboBox()
        self.seg_algo.addItems(['cellpose', 'ilastik', 'farid', 'conv_paint'])
        self.seg_algo.setCurrentIndex(0)
        self.settings_vgroup.glayout.addWidget(QLabel('Algorithm'), 0, 0)
        self.settings_vgroup.glayout.addWidget(self.seg_algo, 0, 1)

        # algo options
        self.cell_diameter = QSpinBox()
        self.cell_diameter.setValue(20)
        self.cell_diameter.setMaximum(10000)
        self.cell_diameter_label = QLabel('Cell diameter')
        self.settings_vgroup.glayout.addWidget(self.cell_diameter_label, 1, 0)
        self.settings_vgroup.glayout.addWidget(self.cell_diameter, 1, 1)

        # smoothing
        self.smoothing = QSpinBox()
        self.smoothing.setMaximum(1000)
        self.smoothing.setValue(1)
        self.settings_vgroup.glayout.addWidget(QLabel('Smoothing'), 2, 0)
        self.settings_vgroup.glayout.addWidget(self.smoothing, 2, 1)

        ## window options
        self.depth = QSpinBox()
        self.depth.setValue(10)
        self.depth.setMaximum(10000)
        self.settings_vgroup.glayout.addWidget(QLabel('Window depth'), 3, 0)
        self.settings_vgroup.glayout.addWidget(self.depth, 3, 1)
        self.width = QSpinBox()
        self.width.setValue(10)
        self.width.setMaximum(10000)
        self.settings_vgroup.glayout.addWidget(QLabel('Window width'), 4, 0)
        self.settings_vgroup.glayout.addWidget(self.width, 4, 1)

        # run analysis
        btn_run = QPushButton("Run analysis")
        btn_run.clicked.connect(self._on_run_analysis)
        self.settings_vgroup.glayout.addWidget(btn_run, 5, 0)
        btn_run_single_segmentation = QPushButton("Run single")
        btn_run_single_segmentation.clicked.connect(self._on_run_seg_spline)
        self.settings_vgroup.glayout.addWidget(btn_run_single_segmentation, 5, 1)
        self.check_use_dask = QCheckBox('Use dask')
        self.check_use_dask.setChecked(False)
        self.settings_vgroup.glayout.addWidget(self.check_use_dask, 5, 2)

        # display options
        self.display_wlayers = QListWidget()
        self.display_wlayers.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.display_wlayers.itemSelectionChanged.connect(self._on_display_wlayers_selection_changed)
        self._display_options_layout.addWidget(QLabel('Window layers'), 0, 0)
        self._display_options_layout.addWidget(self.display_wlayers, 0, 1)

        # dask options
        dask_group = VHGroup('Dask options', 'G')
        dask_group.gbox.setMaximumHeight(150)
        self._dask_layout.addWidget(dask_group.gbox)
        self.dask_num_workers = QSpinBox()
        self.dask_num_workers.setValue(1)
        self.dask_num_workers.setMaximum(64)
        dask_group.glayout.addWidget(QLabel('Number of workers'), 0, 0)
        dask_group.glayout.addWidget(self.dask_num_workers, 0, 1)

        self.dask_cores = QSpinBox()
        self.dask_cores.setValue(1)
        self.dask_cores.setMaximum(64)
        dask_group.glayout.addWidget(QLabel('Number of cores (SLURM)'), 1, 0)
        dask_group.glayout.addWidget(self.dask_cores, 1, 1)

        self.dask_memory = QSpinBox()
        self.dask_memory.setValue(1)
        self.dask_memory.setMaximum(64)
        dask_group.glayout.addWidget(QLabel('Memory per core (SLURM)'), 2, 0)
        dask_group.glayout.addWidget(self.dask_memory, 2, 1)

        self.dask_cluster_type = QComboBox()
        self.dask_cluster_type.addItems(['Local', 'SLURM'])
        self.dask_cluster_type.setCurrentIndex(0)
        dask_group.glayout.addWidget(QLabel('Cluster type'), 3, 0)
        dask_group.glayout.addWidget(self.dask_cluster_type, 3, 1)

        self.dask_initialize_button = QPushButton("Initialize dask")
        self._dask_layout.addWidget(self.dask_initialize_button)
        
        self.dask_stop_cluster_button = QPushButton("Stop dask cluster")
        self._dask_layout.addWidget(self.dask_stop_cluster_button)

        # make sure widgets don't occupy more space than they need
        self._options_layout.addStretch()
        self._paint_layout.addStretch()
        self._dask_layout.addStretch()
        #self._display_options_layout.addStretch()

        self._add_callbacks()

    def _add_callbacks(self):

        self.seg_algo.currentIndexChanged.connect(self._on_update_param)
        self.depth.valueChanged.connect(self._on_update_param)
        self.width.valueChanged.connect(self._on_update_param)
        self.smoothing.valueChanged.connect(self._on_update_param)

        self.segm_channel.currentItemChanged.connect(self._on_update_param)
        self.signal_channel.itemSelectionChanged.connect(self._on_update_param)

        self.btn_load_data.clicked.connect(self._on_load_dataset)

        self.btn_select_analysis.clicked.connect(self._on_click_select_analysis)
        self.btn_select_segmentation.clicked.connect(self._on_click_select_segmentation)

        self.btn_load_analysis.clicked.connect(self._on_load_analysis)

        self.file_list.model().rowsInserted.connect(self._on_change_filelist)
        self.cell_diameter.valueChanged.connect(self._on_update_param)

        self.dask_num_workers.valueChanged.connect(self._on_update_dask_wokers)
        self.dask_initialize_button.clicked.connect(self.initialize_dask)
        self.dask_stop_cluster_button.clicked.connect(self._on_dask_shutdown)

        self.conv_paint_widget.load_model_btn.clicked.connect(self._on_load_model)

    def _on_update_param(self):
        """Update multiple entries of the param object."""
        
        self.param.seg_algo = self.seg_algo.currentText()
        if self.param.seg_algo != 'cellpose':
            self.cell_diameter.setVisible(False)
            self.cell_diameter_label.setVisible(False)
        else:
            self.cell_diameter.setVisible(True)
            self.cell_diameter_label.setVisible(True)
        self.param.lambda_ = self.smoothing.value()
        self.param.width = self.width.value()
        if self.segm_channel.currentItem() is not None:
            self.param.morpho_name = self.segm_channel.currentItem().text()
            self.conv_paint_widget.param.channel = self.param.morpho_name
        if self.signal_channel.currentItem() is not None:
            self.param.signal_name = [x.text() for x in self.signal_channel.selectedItems()]
        if self.file_list.folder_path is not None:
            self.param.data_folder = Path(self.file_list.folder_path)
        if self.display_analysis_folder.text() != 'No selection.':
            self.param.analysis_folder = Path(self.display_analysis_folder.text())
        if self.display_segmentation_folder.text() != 'No selection.':
            self.param.seg_folder = Path(self.display_segmentation_folder.text())
        self.param.diameter = self.cell_diameter.value()

    def _on_click_select_file_folder(self):
        """Interactively select folder to analyze"""

        file_folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.file_list.update_from_path(Path(file_folder))

    def _on_change_filelist(self):
        """Update the channel list when main file list changes."""
        
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        
        self.segm_channel.clear()
        self.signal_channel.clear()
        self.segm_channel.addItems(files)
        self.signal_channel.addItems(files)
        self._on_update_param()

    def _on_load_model(self):
        self.param.random_forest = self.conv_paint_widget.param.random_forest

    def _on_run_analysis(self):
        """Run full morphodynamics analysis"""

        if self.cluster is None and self.check_use_dask.isChecked():
            self.initialize_dask()
        
        model = None
        if self.param.seg_algo == 'conv_paint':
            model = self.load_convpaint_model()

        # run with dask if selected
        if self.check_use_dask.isChecked():
            with Client(self.cluster) as client:
                self.res = analyze_morphodynamics(
                    data=self.data,
                    param=self.param,
                    client=client,
                    model=model
                )
        else:
            self.res = analyze_morphodynamics(
                    data=self.data,
                    param=self.param,
                    client=None,
                    model=model
                )
        
        self._on_load_windows()
        export_results_parameters(self.param, self.res)

    def _on_run_seg_spline(self):

        if self.param.seg_algo == 'conv_paint':
            self.load_convpaint_model(return_model=False)
        step = self.viewer.dims.current_step[0]
        image, c, im_windows, windows = compute_spline_windows(self.param, step)

        layer_indices= self._get_layer_indices(windows)
        col_dict, _ = self._create_color_shadings(layer_indices)

        self.viewer.add_labels(image, name='segmentation')
        self.viewer.add_labels(im_windows, name='windows')
        self.viewer.layers['windows'].color = col_dict
        self.viewer.layers['windows'].color_mode = 'direct' 
        self.viewer.add_shapes(
            data=[np.c_[c[1], c[0]]], shape_type='polygon', 
            edge_color='red', face_color=[0,0,0,0], edge_width=1,
            name='spline')


    def _on_segment_single_frame(self):
        """Segment single frame."""
        
        step = self.viewer.dims.current_step[0]
        temp_image = segment_single_frame(
            self.param, step, self.param.analysis_folder, return_image=True)
        
        self.viewer.add_labels(temp_image)
        #self.viewer.open(self.param.analysis_folder.joinpath("segmented_k_" + str(step) + ".tif"))

    def initialize_dask(self, event=None):
        """Initialize dask client.
        To do: add SLURMCluster and and interface for it"""

        if self.dask_cluster_type.currentText() == 'Local':
            self.cluster = LocalCluster()#n_workers=self.dask_num_workers.value())
            self.dask_num_workers.setValue(len(self.cluster.scheduler_info['workers']))
        elif self.dask_cluster_type.currentText() == 'SLURM':
            self.cluster = SLURMCluster(cores=self.dask_cores.value(), memory=self.dask_memory.value())

    def _on_update_dask_wokers(self):
        """Update dask workers."""
        
        if self.dask_cluster_type.currentText() == 'Local':
            if self.cluster is None:
                self.initialize_dask()
                self.cluster = LocalCluster(n_workers=self.dask_num_workers.value())

            self.cluster.scale(self.dask_num_workers.value())

    def _on_dask_shutdown(self):
        """Shutdown dask workers."""
        
        self.cluster.close()
        self.cluster = None

    def _on_click_select_analysis(self):
        """Select folder where to save the analysis."""

        self.analysis_path = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        self.display_analysis_folder.setText(self.analysis_path.as_posix())
        self._on_update_param()
        if self.param.seg_folder is None:
            self.param.seg_folder = self.analysis_path.joinpath('main_segmentation')
            self.display_segmentation_folder.setText(self.param.seg_folder.as_posix())

    def _on_click_select_segmentation(self):
        """Select folder where to save the segmentation."""

        self.segmentation_path = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        self.display_segmentation_folder.setText(self.segmentation_path.as_posix())
        self._on_update_param()

    def load_convpaint_model(self, return_model=True):
        """Load RF model for segmentation"""
        
        #if self.conv_paint_widget.random_forest is not None:
        #    self.conv_paint_widget.save_model()
        #else:
        if self.conv_paint_widget.random_forest is None:
            self.conv_paint_widget.load_model()
        self.param.random_forest = self.conv_paint_widget.param.random_forest
        model=self.conv_paint_widget.random_forest
        if return_model:
            return model
        else:
            return None

    def _on_load_dataset(self):
        """Having selected segmentation and signal channels, load the data"""
        
        self.data, self.param = dataset_from_param(self.param)
        self.create_stacks()

    def _on_display_wlayers_selection_changed(self):
        """Hide/reveal window layers."""
        
        on_list = [int(x.text()) for x in self.display_wlayers.selectedItems()]
        for i in self.layer_indices:
            if i not in on_list:
                val_to_set = 0
            else:
                val_to_set = 1
            for j in self.layer_global_indices[i]:
                self.viewer.layers['windows'].color[j][-1]=val_to_set
        self.viewer.layers['windows'].color_mode = 'direct'

    def create_stacks(self):
        """Create and add to the viewer datasets as dask stacks.
        Note: this should be added to the dataset class."""

        sample = self.data.load_frame_morpho(0)

        if self.data.data_type == 'h5':
            h5file = self.data.channel_imobj[self.data.channel_name.index(self.param.morpho_name)]
            seg_stack = da.from_array(h5file)
        else:
            my_data = self.data
            def return_image(i):
                return my_data.load_frame_morpho(i)
            seg_lazy_arrays = [delayed(return_image)(i) for i in range(self.data.max_time)]
            dask_seg_arrays = [da.from_delayed(x, shape=sample.shape, dtype=sample.dtype) for x in seg_lazy_arrays]
            seg_stack = da.stack(dask_seg_arrays, axis=0)
        self.viewer.add_image(seg_stack, name=self.param.morpho_name)

        for ind, c in enumerate(self.param.signal_name):
            if self.data.data_type == 'h5':
                h5file = self.data.channel_imobj[self.data.channel_name.index(c)]
                sig_stack = da.from_array(h5file)
            else:
                my_data = self.data
                def return_image(ch, t):
                    return my_data.load_frame_signal(ch, t)

                sig_lazy_arrays = [delayed(return_image)(ind, i) for i in range(self.data.max_time)]
                dask_sig_arrays = [da.from_delayed(x, shape=sample.shape, dtype=sample.dtype) for x in sig_lazy_arrays]
                sig_stack = da.stack(dask_sig_arrays, axis=0)
            self.viewer.add_image(sig_stack, name=f'signal {c}')
    
    def create_stacks_classical(self):
        """Create stacks from the data and add them to viewer."""
        
        seg_stack = np.stack(
            [self.data.load_frame_morpho(i) for i in range(self.data.max_time)], axis=0)
        self.viewer.add_image(seg_stack, name=self.param.morpho_name)

        sig_stack = [np.stack(
            [self.data.load_frame_signal(c, i) for i in range(self.data.max_time)], axis=0)
            for c in range(len(self.param.signal_name))]
        for ind, c in enumerate(self.param.signal_name):
            self.viewer.add_image(sig_stack[ind], name=f'signal {c}')

    def _get_layer_indices(self, windows):
        """Given a windows list of lists, create a dictionary where each entry i
        contains the labels of all windows in layer i"""

        layer_indices= {}
        count=1
        for i in range(len(windows)):
            gather_indices=[]
            for j in range(len(windows[i])):
                gather_indices.append(count)
                count+=1
            layer_indices[i]=gather_indices
        return layer_indices

    def _create_color_shadings(self, layer_index):
        """Given a layer index (from _get_layer_indices), create a dictionary where
        each entry i contains the colors for label i. Colors are in shades of a given
        color per layer."""

        ## create a color dictionary, containing for each label index a color
        ## currently each layer gets a color and labeles within it get a shade of that color
        color_layers = ['red', 'blue', 'cyan', 'magenta']
        color_pool = cycle(color_layers)
        global_index=1
        col_dict = {None: np.array([0., 0., 0., 1.], dtype=np.float32)}
        layer_global_indices = {i: [] for i in range(len(layer_index))}
        for (lay, col) in zip(layer_index, color_pool):
            num_colors = len(layer_index[lay])
            #color_array = napari.utils.colormaps.SIMPLE_COLORMAPS[col].map(np.linspace(0.1,1,num_colors))
            color_array = cycle(napari.utils.colormaps.SIMPLE_COLORMAPS[col].map(np.array([0.3,0.45,0.6,0.75, 0.9])))
            for ind2, col_index in zip(range(num_colors), color_array):
                col_dict[global_index]=col_index
                layer_global_indices[lay].append(global_index)
                global_index+=1
        return col_dict, layer_global_indices

    def _on_load_windows(self):
        """Add windows labels to the viewer"""

        # create array to contain the windows
        w_image = np.zeros((
            self.data.max_time,
            self.viewer.layers[self.param.morpho_name].data.shape[1],
            self.viewer.layers[self.param.morpho_name].data.shape[2]), dtype=np.uint16)
        
        # load window indices and use them to fill window array
        # keep track of layer to which indices belong in self.layer_indices
        for t in range(self.data.max_time):
            name = Path(self.param.analysis_folder).joinpath(
                'segmented', "window_k_" + str(t) + ".pkl")
            windows = pickle.load(open(name, 'rb'))
            if t==0:
                self.layer_indices= self._get_layer_indices(windows)

            w_image[t, :, :] = label_windows(
                shape=(self.data.shape[1], self.data.shape[2]), windows=windows)

        col_dict, self.layer_global_indices = self._create_color_shadings(self.layer_indices)
        
        # assign color dictionary to window layer colormap
        self.viewer.add_labels(w_image, name='windows')
        self.viewer.layers['windows'].color = col_dict
        self.viewer.layers['windows'].color_mode = 'direct' #needed to refresh the color map

        self.display_wlayers.addItems([str(x) for x in self.layer_indices.keys()])

    def _on_load_analysis(self):
        """Load existing output of analysis"""
        
        if self.analysis_path is None:
            self._on_click_select_analysis()

        self.param, self.res, self.data = load_alldata(
            self.analysis_path, load_results=True
        )
        self._on_update_interface()
        self.file_list.update_from_path(self.param.data_folder)
        self.create_stacks()
        self._on_load_windows()
        

    def _on_update_interface(self):
        """Update UI when importing existing analyis"""

        self.seg_algo.setCurrentText(self.param.seg_algo)
        self.cell_diameter.setValue(self.param.diameter)

def scroll_label(default_text = 'default text'):
    mylabel = QLabel()
    mylabel.setText('No selection.')
    myscroll = QScrollArea()
    myscroll.setWidgetResizable(True)
    myscroll.setWidget(mylabel)
    return mylabel, myscroll