import matplotlib.pyplot as plt
from skimage.external.tifffile import imsave

class Plot:
    def __init__(self, debug):
        self.n = 0
        self.debug = debug
        self.path = 'C:/Work/UniBE2/Code/Output/'

    def imshow(self, title, image, nfig=None, cmap=None, cbar=False):
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
            # plt.savefig(self.path + str(self.n) + ' ' + title + '.svg')
            # plt.savefig(self.path + str(self.n) + ' ' + title + '.eps')
            # plt.get_current_fig_manager().window.move(960, -1080)
            plt.get_current_fig_manager().window.showMaximized()

    def plotopen(self, title, nfig=None):
        if nfig == None:
            self.n += 1
        else:
            self.n = nfig
        self.title = title
        plt.figure(self.n)
        plt.clf()
        plt.gcf().suptitle(title)

    def plotclose(self, save=True):
        if save:
            plt.savefig(self.path + str(self.n) + ' ' + self.title + '.pdf')
            # plt.savefig(self.path + str(self.n) + ' ' + self.title + '.svg')
            # plt.savefig(self.path + str(self.n) + ' ' + self.title + '.eps')
        # px = plt.get_current_fig_manager().canvas.width()
        # py = plt.get_current_fig_manager().canvas.height()
        # plt.get_current_fig_manager().window.move(960, -1080)
        plt.get_current_fig_manager().window.showMaximized()

    def show(self):
        if self.debug:
            plt.show()