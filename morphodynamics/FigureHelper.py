import matplotlib.pyplot as plt
from skimage.external.tifffile import imsave
from matplotlib.backends.backend_pdf import PdfPages
from skimage.external.tifffile import TiffWriter

# PDF
# Figure window
# Figure in notebook
# TIFF: entire figure or just image


class FigureHelper:
    """ Helper class for generating figures, when in debug mode. """

    def __init__(self, name, output):
        self.name = name
        self.output = output
        self.n = 0

    # def imshow(self, title, image, nfig=None, cmap=None, cbar=False):
    #     """ A replacement for matplotlib.pyplot.imshow that takes care of titles, saving figures, etc. """
    #     if self.output.window:
    #         if nfig is None:
    #             self.n += 1
    #         else:
    #             self.n = nfig
    #         if cmap is None:
    #             cmap = 'gray'
    #         plt.figure(self.n).suptitle(title)
    #         plt.imshow(image, cmap=cmap)
    #         if cbar:
    #             plt.colorbar()
    #         imsave(self.output.dir + str(self.n) + ' ' + title + '.tif', image, compress=6)
    #         plt.savefig(self.output.dir + str(self.n) + ' ' + title + '.pdf')
    #         # plt.get_current_fig_manager().window.move(960, -1080)
    #         plt.get_current_fig_manager().window.showMaximized()

    def open_figure(self, title, nfig=None, figure_size=None):
        """ Initialize a new figure. """
        if nfig is None:
            self.n += 1
        else:
            self.n = nfig
        if figure_size is None:
            figure_size = self.output.size
        # self.name = title
        plt.figure(self.n).set_size_inches(figure_size)
        # plt.get_current_fig_manager().window.showMaximized()
        plt.clf()
        plt.tight_layout()
        plt.gca().set_title(title)

    def close_figure(self):
        """ Save a figure. """
        if self.output.pdf:
            plt.savefig(self.output.dir + str(self.n) + ' ' + self.name + '.pdf')

    def save_pdf(self):
        if not hasattr(self, 'pp'):
            self.pp = PdfPages(self.output.dir + self.name + '.pdf')
        if self.output.pdf:
            self.pp.savefig()

    def save_tiff(self, x, compress=6):
        if not hasattr(self, 'tw'):
            self.tw = TiffWriter(self.output.dir + self.name + '.tif')
        if self.output.tiff:
            self.tw.save(x, compress=compress)

    def show(self):
        if self.output.display:
            plt.show()

    def close(self):
        self.show()
        if hasattr(self, 'pp') & self.output.pdf:
            self.pp.close()
        if hasattr(self, 'tw') & self.output.tiff:
            self.tw.close()
