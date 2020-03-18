import matplotlib.pyplot as plt
from skimage.external.tifffile import imsave
import time

class FigureHelper:
    """ Helper class for generating figures, when in debug mode. """
    def __init__(self, debug):
        self.n = 0
        self.debug = debug
        # self.path = 'C:/Work/UniBE2/Code/Output/'
        self.path = './'

    def imshow(self, title, image, nfig=None, cmap=None, cbar=False):
        """ A replacement for matplotlib.pyplot.imshow that takes care of titles, saving figures, etc. """
        if self.debug:
            if nfig == None:
                self.n += 1
            else:
                self.n = nfig
            if cmap == None:
                cmap = 'gray'
            plt.figure(self.n).suptitle(title)
            plt.imshow(image, cmap=cmap)
            if cbar:
                plt.colorbar()
            imsave(self.path + str(self.n) + ' ' + title + '.tif', image, compress=6)
            plt.savefig(self.path + str(self.n) + ' ' + title + '.pdf')
            # plt.get_current_fig_manager().window.move(960, -1080)
            plt.get_current_fig_manager().window.showMaximized()

    def openFigure(self, title, nfig=None, figsize=(16, 9)):
        """ Initialize a new figure. """
        if nfig == None:
            self.n += 1
        else:
            self.n = nfig
        self.title = title
        plt.figure(self.n).set_size_inches(figsize)
        plt.get_current_fig_manager().window.showMaximized()
        plt.clf()
        plt.tight_layout()
        plt.gca().set_title(title)

    def closeFigure(self):
        """ Save a figure. """
        if self.debug:
            plt.savefig(self.path + str(self.n) + ' ' + self.title + '.pdf')

    def show(self):
        if self.debug:
            plt.show()

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))