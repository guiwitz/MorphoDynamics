import dill
import os
from pathlib import Path

from morphodynamics.folders import Folders
from morphodynamics.plots.show_plots import *
from morphodynamics.plots.ui_edge_vectorial import EdgeVectorialSlow
from morphodynamics.plots.ui_edge_rasterized import EdgeRasterized
from morphodynamics.plots.ui_curvature import Curvature
from morphodynamics.plots.ui_signals import Signals
from morphodynamics.plots.ui_correlation import Correlation
from morphodynamics.plots.ui_batch import BatchExport


import morphodynamics.figurehelper
from morphodynamics.settings import Struct, load_settings
from morphodynamics.utils import load_alldata
import matplotlib.pyplot as plt
import ipywidgets as ipw

from IPython.display import display, HTML


# suppress figure titles in widgets rendering and enlarge notebook
display(
    HTML("<style>div.jupyter-widgets.widget-label {display: none;}</style>")
)
display(HTML("<style>.container { width:100% !important; }</style>"))


class OutputUI:
    def __init__(
        self,
        expdir=None,
    ):

        style = {"description_width": "initial"}
        layout = {"width": "300px"}

        self.expdir = expdir

        self.result_folder = Folders(window_width=300)
        self.result_folder.file_list.observe(
            self.update_dir, names=("options", "value")
        )

        if self.expdir is not None:
            self.expdir = Path(self.expdir)
            self.result_folder.cur_dir = self.expdir
            self.result_folder.refresh(None)

        # run the analysis button
        self.load_button = ipw.Button(description="Click to load")
        self.load_button.on_click(self.load_data)

        self.tab = ipw.Tab()
        self.batch_out = ipw.Output()

        self.interface = ipw.VBox(
            [
                self.result_folder.file_list,
                self.load_button,
                self.tab,
                ipw.HTML(
                            '<font size="5"><b>Batch export<b></font>'
                        ),
                self.batch_out,
            ]
        )

    def load_data(self, b=None):

        self.param, self.res, self.data = load_alldata(
            self.expdir, load_results=True
        )
        self.create_ui()

        self.batch = BatchExport(self.param, self.data, self.res)
        self.batch.create_interface()
        with self.batch_out:
            display(self.batch.interface)

    def update_dir(self, change):

        self.expdir = self.result_folder.cur_dir

    def create_ui(self):

        self.out1 = ipw.Output()
        self.names = []
        self.outputs = []
        with self.out1:
            self.fig1, _ = show_circularity(
                self.param, self.data, self.res, size=(8, 3)
            )

        self.outputs.append(self.out1)
        self.names.append("Circularity")

        self.out2 = ipw.Output()
        with self.out2:
            self.fig2, _ = show_edge_overview(
                self.param, self.data, self.res, lw=0.3, size=(8, 6)
            )
        self.outputs.append(self.out2)
        self.names.append("Edge overview")

        self.ev = EdgeVectorialSlow(self.param, self.data, self.res)
        self.ev.create_interface()
        self.outputs.append(self.ev.interface)
        self.names.append("Interactive edge (vectorial)")

        self.er = EdgeRasterized(self.param, self.data, self.res)
        self.er.create_interface()
        self.outputs.append(self.er.interface)
        self.names.append("Interactive edge (rasterized)")

        self.curvature = Curvature(self.param, self.data, self.res)
        self.curvature.create_interface()
        self.outputs.append(self.curvature.interface)
        self.names.append("Curvature")

        self.out3 = ipw.Output()
        with self.out3:
            self.fig3, _ = show_displacement(
                self.param, self.res, size=(8, 4.5)
            )
        self.outputs.append(self.out3)
        self.names.append("Displacement")

        self.out4 = ipw.Output()
        with self.out4:
            self.fig4, _ = show_cumdisplacement(
                self.param, self.res, size=(8, 4.5)
            )
        self.outputs.append(self.out4)
        self.names.append("Cumul. Displacement")

        self.s = Signals(self.param, self.data, self.res)
        self.s.create_interface()
        self.outputs.append(self.s.interface)
        self.names.append("Signals")

        self.c = Correlation(self.param, self.data, self.res)
        self.c.create_interface()
        self.c.interface
        self.outputs.append(self.c.interface)
        self.names.append("Correlation")

        self.tab.children = self.outputs

        for ind, x in enumerate(self.outputs):
            self.tab.set_title(ind, self.names[ind])
