import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as ipw
import pickle
import skimage.io
import skimage.morphology
import skimage.segmentation
from skimage.measure import find_contours
import matplotlib
cmap2 = matplotlib.colors.ListedColormap (np.array([[1,0,0,0.5],[1,0,0,0.5]]))
from IPython.display import display

from ..windowing import (
    calculate_windows_index,
    create_windows,
    boundaries_image,
)

class Wimage:
    """
    Interactive UI to visualize the edge displacement over time"""

    def __init__(self, param, data, res):
        self.param = param
        self.data = data
        self.res = res

        self.out = ipw.Output()
        
        if self.data is not None:
            image = self.data.load_frame_morpho(0)
            channel_list = [self.param.morpho_name]+self.param.signal_name
            max_time = self.data.num_timepoints-1
        else:
            image = np.zeros((3,3))
            channel_list = []
            max_time = 0

        self.display_channel_ipw = ipw.Select(options=channel_list)
        self.display_channel_ipw.observe(self.show_segmentation, names="value")

        # slider for time limits to analyze
        self.time_slider = ipw.IntSlider(
            description="Time", min=0, max=max_time, value=0, continuous_update=True
        )
        self.time_slider.observe(self.show_segmentation, names="value")

        # intensity sliders
        self.intensity_range_slider = ipw.IntRangeSlider(
            description="Intensity range",
            min=0,
            max=0,
            value=(0, 0),
            continuous_update=True,
        )
        self.intensity_range_slider.observe(self.update_intensity_range, names="value")

        # show windows or nots
        self.show_windows_choice = ipw.Checkbox(description="Show windows", value=True)
        self.show_windows_choice.observe(self.update_windows_vis, names="value")

        # show text or not
        self.show_text_choice = ipw.Checkbox(description="Show labels", value=True)
        self.show_text_choice.observe(self.update_text_vis, names="value")

        #with self.out:
        # https://github.com/matplotlib/ipympl/issues/366#issuecomment-937519285
        with plt.ioff():    
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.ax.set_title(f"Time:")

            self.implot = self.ax.imshow(np.zeros((2, 2)), cmap='gray')
            self.wplot = None#self.ax.imshow(np.zeros((20,20)), cmap=cmap2)
            self.tplt = None
            self.cplot = None
            #display(self.fig.canvas)

            #self.fig.show()

        self.interface = ipw.HBox([
            #self.out,
            self.fig.canvas,
            ipw.VBox([
                self.time_slider, self.intensity_range_slider, self.display_channel_ipw,
                self.show_windows_choice, self.show_text_choice])
            ])

    def update_dataset(self, param, data, res):

        self.param = param
        self.res = res
        self.data = data

        self.display_channel_ipw.options = [self.param.morpho_name]+self.param.signal_name
        self.time_slider.max = self.data.num_timepoints-1

    def load_image(self, time):
        """Load image selected in self.display_channel_ipw widget"""

        if self.display_channel_ipw.value == self.param.morpho_name:
            image = self.data.load_frame_morpho(time)
        else:
            channel_index = self.param.signal_name.index(self.display_channel_ipw.value)
            image = self.data.load_frame_signal(channel_index, time)

        return image

    def update_intensity_range(self, change=None):
        """Callback to update intensity range"""

        self.intensity_range_slider.max = self.implot.get_array().max()
        self.implot.set_clim(vmin=self.intensity_range_slider.value[0], vmax=self.intensity_range_slider.value[1])
        self.fig.canvas.draw_idle()

    def update_windows_vis(self, change=None):
        """Callback to turn windows visibility on/off"""
        if self.cplot[0][0].get_alpha() == 0:
            turnonoff = [t[0].set_alpha(1) for t in self.cplot]
        else:
            turnonoff = [t[0].set_alpha(0) for t in self.cplot]
        self.fig.canvas.draw_idle()

    def update_text_vis(self, change=None):
        """Callback to turn windows labels visibility on/off"""

        if self.tplt[0].get_alpha() == 0:
            turnonoff = [t.set_alpha(1) for t in self.tplt]
        else:
            turnonoff = [t.set_alpha(0) for t in self.tplt]
        self.fig.canvas.draw_idle()
    
    def show_segmentation(self, change=None):
        """Update segmentation plot"""

        t = self.time_slider.value
        # load image of channel selected in self.display_channel_ipw
        if self.data is not None:
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
                im_w = skimage.io.imread(name)
                b0 = im_w.astype(float)
                b0[b0 == 0] = np.nan
                windows_pos = np.array(calculate_windows_index(window))

                # find contours
                im_lab = skimage.morphology.label(1-skimage.morphology.skeletonize(im_w),connectivity=1)
                im_lab[im_lab==im_lab[0,0]]=0
                im_expand = skimage.segmentation.expand_labels(im_lab,distance=3)
                contours = []
                for i in np.unique(im_expand):
                    if i>0:
                        temp = im_expand == i
                        contours.append(find_contours(temp,level=0.5)[0])

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
                    '''if self.wplot is None:
                        self.wplot = self.ax.imshow(b0, cmap=cmap2)
                        self.wplot.set_extent((0, image.shape[1], 0, image.shape[0]))
                    else:
                        self.wplot.set_array(b0)'''

                    if self.cplot is None:
                        self.cplot = [self.ax.plot(c[:,1], image.shape[0]-c[:,0],'r') for c in contours]
                    else:
                        for ind, c in enumerate(contours):
                            if ind < len(self.cplot):
                                self.cplot[ind][0].set_data((c[:,1],image.shape[0]-c[:,0]))
                            else:
                                self.cplot.append(self.ax.plot(c[:,1], image.shape[0]-c[:,0],'r'))
                        for ind in range(len(contours), len(self.cplot)):
                            self.cplot[ind][0].set_data(([],[]))

                if windows_pos is not None:
                    if self.tplt is None:
                        self.tplt = [self.ax.text(
                            p[0],
                            image.shape[0]-p[1],
                            int(p[2]),
                            color="red",
                            fontsize=10,
                            horizontalalignment="center",
                            verticalalignment="center",
                        ) for p in windows_pos]
                    else:
                        for ind, p in enumerate(windows_pos):
                            self.tplt[ind].set_x(p[0])
                            self.tplt[ind].set_y(image.shape[0]-p[1])
                            self.tplt[ind].set_text(str(int(p[2])))
        self.fig.canvas.draw_idle()