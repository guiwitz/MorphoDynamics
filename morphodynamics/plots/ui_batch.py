import os
import matplotlib.pyplot as plt
from .show_plots import (
    show_circularity,
    show_edge_overview,
    save_edge_vectorial_movie,
    show_displacement,
    show_cumdisplacement,
    show_curvature,
    save_signals,
)
from .ui_edge_rasterized import EdgeRasterized

import ipywidgets as ipw


def show_analysis(data, param, res):
    """ Display the results of the morphodynamics analysis. """

    if param.showCircularity:
        # with out:
        fig, ax = show_circularity(param, data, res)
        fig.savefig(os.path.join(param.analysis_folder, "Circularity.png"))
        plt.close()

    if param.showEdgeOverview:
        # with out:
        fig, ax = show_edge_overview(param, data, res)
        fig.savefig(os.path.join(param.analysis_folder, "Edge_overview.png"))
        plt.close()

    if param.showEdgeVectorial:
        save_edge_vectorial_movie(param, data, res, curvature=False)

    if param.showDisplacement:
        fig, ax = show_displacement(param, res)
        fig.savefig(os.path.join(param.analysis_folder, "Displacement.png"))
        plt.close()

        fig, ax = show_cumdisplacement(param, res)
        fig.savefig(os.path.join(param.analysis_folder, "Cumul_Displacement.png"))
        plt.close()

    if param.showEdgeRasterized:
        er = EdgeRasterized(param, data, res)
        er.save_movie("border")
        er.save_movie("curvature")
        er.save_movie("displacement")
        er.save_movie("cumulative displacement")
        er.save_movie("cumulative displacement")

    if param.showCurvature:
        fig, ax = show_curvature(param, data, res)
        fig.savefig(os.path.join(param.analysis_folder, "Curvature.png"))
        plt.close()

    if param.showSignals:
        save_signals(param, data, res)

    """
    if param.showCorrelation:
        show_correlation(param, data, res)

    if param.showFourierDescriptors:
        show_fourier_descriptors(param, data, res)"""


class BatchExport:
    def __init__(self, param, data, res):
        self.param = param
        self.data = data
        self.res = res
        self.out = ipw.Output()

    def create_interface(self):

        circularity_checkbox = ipw.Checkbox(
            value=self.param.showCircularity,
            description="Circularity",
            disabled=False,
            indent=False,
        )

        edge_overview_checkbox = ipw.Checkbox(
            value=self.param.showEdgeOverview,
            description="Edge overview",
            disabled=False,
            indent=False,
        )

        edge_vectorial_checkbox = ipw.Checkbox(
            value=self.param.showEdgeVectorial,
            description="Edge vectorial",
            disabled=False,
            indent=False,
        )

        edge_rasterized_checkbox = ipw.Checkbox(
            value=self.param.showEdgeRasterized,
            description="Edge rasterized",
            disabled=False,
            indent=False,
        )

        curvature_checkbox = ipw.Checkbox(
            value=self.param.showCurvature,
            description="Curvature",
            disabled=False,
            indent=False,
        )

        displacement_checkbox = ipw.Checkbox(
            value=self.param.showDisplacement,
            description="Displacement",
            disabled=False,
            indent=False,
        )

        signals_checkbox = ipw.Checkbox(
            value=self.param.showSignals,
            description="Signals",
            disabled=False,
            indent=False,
        )

        """correlation_checkbox = ipw.Checkbox(
            value=self.param.showCorrelation,
            description="Correlation",
            disabled=False,
            indent=False,
        )

        fourier_descriptors_checkbox = ipw.Checkbox(
            value=self.param.showFourierDescriptors,
            description="Fourier descriptors",
            disabled=False,
            indent=False,
        )"""

        export_button = ipw.Button(
            description="Export figures",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            # tooltip='Click me',
            # icon='check'  # (FontAwesome names without the `fa-` prefix)
        )

        def export_figures(change):
            with self.out:
                self.param.showCircularity = circularity_checkbox.value
                self.param.showEdgeOverview = edge_overview_checkbox.value
                self.param.showEdgeVectorial = edge_vectorial_checkbox.value
                self.param.showEdgeRasterized = edge_rasterized_checkbox.value
                self.param.showCurvature = curvature_checkbox.value
                self.param.showDisplacement = displacement_checkbox.value
                self.param.showSignals = signals_checkbox.value
                """self.param.showCorrelation = correlation_checkbox.value
                self.param.showFourierDescriptors = (
                    fourier_descriptors_checkbox.value
                )
                """
                # matplotlib.use("PDF")
                show_analysis(self.data, self.param, self.res)
                # matplotlib.use("nbAgg")

        export_button.on_click(export_figures)

        self.interface = ipw.VBox(
            [
                circularity_checkbox,
                edge_overview_checkbox,
                edge_vectorial_checkbox,
                edge_rasterized_checkbox,
                curvature_checkbox,
                displacement_checkbox,
                signals_checkbox,
                # correlation_checkbox,
                # fourier_descriptors_checkbox,
                export_button,
            ]
        )
