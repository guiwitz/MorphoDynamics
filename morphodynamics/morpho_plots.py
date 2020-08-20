import numpy as np
import matplotlib.pyplot as plt


def show_windows_set(image, w, b, windows_pos, ax=None, implot=None, wplot=None, tplt=None):

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
        fig.tight_layout()

    if implot is None:
        implot = ax.imshow(image, cmap = 'gray')
    else:
        implot.set_data(image)

    if wplot is None:
        wplot = ax.imshow(b, cmap = 'Greens', vmin = 0, vmax=2)
    else:
        wplot.set_data(b)
    
    if tplt is None:
        tplt = []
        #num_w = np.sum([len(x) for x in w])
        for p in windows_pos:
            tplt.append(ax.text(p[0], p[1], p[2], color='yellow', fontsize=8, horizontalalignment='center', verticalalignment='center'))
    else:
        for ind, p in enumerate(windows_pos):
            tplt[ind].set_x(p[0])
            tplt[ind].set_y(p[1])
            tplt[ind].set_text(str(p[2]))

    return fig, ax, implot, wplot, tplt


def show_windows(image, b, windows_pos):
    #fig, ax = plt.subplots(figsize=(5,5))

    plt.clf()
    ax = plt.axes()
    implot = ax.imshow(image, cmap = 'gray')
    wplot = ax.imshow(b, cmap = 'Greens', vmin = 0, vmax=2)

    tplt = []
    for p in windows_pos:
        tplt.append(ax.text(p[0], p[1], p[2], color='yellow', fontsize=8, horizontalalignment='center', verticalalignment='center'))

    return implot, wplot, tplt
