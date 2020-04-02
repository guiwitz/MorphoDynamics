import matplotlib.pyplot as plt
from skimage.external.tifffile import imsave

class FigureHelper:
    """ Helper class for generating figures, when in debug mode. """
    def __init__(self, debug):
        self.n = 0
        self.debug = debug
        # self.path = 'C:/Work/UniBE2/Code/Output/'
        self.path = './'
        self.title = ''

    def imshow(self, title, image, nfig=None, cmap=None, cbar=False):
        """ A replacement for matplotlib.pyplot.imshow that takes care of titles, saving figures, etc. """
        if self.debug:
            if nfig is None:
                self.n += 1
            else:
                self.n = nfig
            if cmap is None:
                cmap = 'gray'
            plt.figure(self.n).suptitle(title)
            plt.imshow(image, cmap=cmap)
            if cbar:
                plt.colorbar()
            imsave(self.path + str(self.n) + ' ' + title + '.tif', image, compress=6)
            plt.savefig(self.path + str(self.n) + ' ' + title + '.pdf')
            # plt.get_current_fig_manager().window.move(960, -1080)
            plt.get_current_fig_manager().window.showMaximized()

    def open_figure(self, title, nfig=None, figsize=(16, 9)):
        """ Initialize a new figure. """
        if nfig is None:
            self.n += 1
        else:
            self.n = nfig
        self.title = title
        plt.figure(self.n).set_size_inches(figsize)
        # plt.get_current_fig_manager().window.showMaximized()
        plt.clf()
        plt.tight_layout()
        plt.gca().set_title(title)

    def close_figure(self):
        """ Save a figure. """
        if self.debug:
            plt.savefig(self.path + str(self.n) + ' ' + self.title + '.pdf')

    def show(self):
        if self.debug:
            plt.show()
