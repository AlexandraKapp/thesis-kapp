# @Patrick Siegler
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import numpy as np

import conf
import utils

clmap = np.tile(np.array(['r', 'g', 'b', 'c', 'y', 'm', 'gray', 'black', 'orange', 'lime', 'violet', 'maroon', 'darkolivegreen', 'royalblue', 'sandybrown', 'forestgreen', 'lightpink', 'deepskyblue', 'gold', 'tomato', 'indigo', 'steelblue', 'yellow', 'goldenrod'], dtype=object), 30)
clmap_lastnoise = np.copy(clmap)
clmap_lastnoise[len(clmap_lastnoise)-1] = 'lightgray'
clmap_firstnoise = np.insert(clmap, 0, 'lightgray')


def _finalize(name):
    if conf.plotFile:
        name = conf.plotDir + '/' + name
        utils.makedirs(name)
        plt.savefig(name + '.pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.gcf().canvas.set_window_title(name)


def plot_chartmatrix(lines, name, labels = None):
    print('plotting chart matrix')
    fig, axes = plt.subplots(conf.gridXmax - conf.gridXmin + 1, conf.gridYmax - conf.gridYmin + 1)
    fig.set_size_inches(40, 15)
    coloridx = 0
    for i in range(conf.gridXmin, conf.gridXmax+1):
        for j in range(conf.gridYmin, conf.gridYmax+1):
            try:
                sp = lines[(i-1)*conf.gridXdim+j]
                if labels is not None:
                    pl = sp.plot(ax=axes[conf.gridXmax-i,j-conf.gridYmin], label=str((i-1)*conf.gridXdim+j), c=clmap_lastnoise[labels[coloridx]])
                else:
                    pl = sp.plot(ax=axes[conf.gridXmax-i,j-conf.gridYmin], label=str((i-1)*conf.gridXdim+j))
                if conf.zscore:
                    pl.set_ylim(-1.0,3.0)
                coloridx = coloridx + 1
            except KeyError:
                pass

    _finalize(name)


def plot_pixmap(labels, name):
    print('plotting pixmap')
    plt.figure()
    plt.imshow(labels.reshape((conf.gridXmax - conf.gridXmin + 1, conf.gridYmax - conf.gridYmin + 1)),
               interpolation='None',
               cmap=pltcol.ListedColormap(clmap_firstnoise, N=max(labels)+2),
               vmin=-1,
               vmax=max(labels)+1)
    #plt.colorbar()
    plt.gca().invert_yaxis()

    _finalize(name)


def plot_corrmat(dist, name, hardscale=True):
    print('plotting corrmat')
    plt.figure()
    if hardscale:
        plt.imshow(dist,
               interpolation='None',
               vmin=-1,
               vmax=1)
    else:
        plt.imshow(dist,
               interpolation='None',
               vmin=-0.5,
               vmax=0.5)
    plt.colorbar()
    plt.gcf().set_size_inches(20, 15)

    _finalize(name)


def plot_hist(df, name, bins=20, range=None):
    print('plotting histogram')
    df.plot.hist(bins, range=range)
    _finalize(name)
