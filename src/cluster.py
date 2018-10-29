#@Patrick Siegler

import matplotlib.pyplot as plt
from utils import tqdm
import numpy as np

import conf
import utils
import loader
import distance
import plotting
import fast_dbscan
from performance import PhaseTimer
from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.externals.joblib import Parallel, delayed


def cluster(p, dist, t, eps, plot=False):
    if conf.clustering == 'DBSCAN':
        scan = DBSCAN(eps=eps, min_samples=conf.minPts, metric='precomputed', n_jobs=conf.nJobs)
        labels = scan.fit_predict(dist)
    elif conf.clustering == 'FastDBSCAN':
        labels = np.asarray(fast_dbscan.dbscan(dist, eps, conf.minPts))
        labels = labels - 1
    elif conf.clustering == 'COREQ':
        import BlockCorr
        labels = BlockCorr.Cluster(p.transpose(), eps, conf.minPts, 0)
        labels = labels - 1
        distance.distances += BlockCorr.CorrCount()
    else:
        raise NotImplementedError

    # print(labels[:100])
    if conf.verbose:
        print('{:.5f}: {} clusters covering {} series found'.format(eps, max(labels) + 1, np.count_nonzero(labels != -1)))

    if plot:
        plotting.plot_pixmap(labels, 'cluster/pixmap_{}_w{}_eps{:.3f}_{}'.format(conf.data, conf.wnd, eps, t.strftime('%Y-%m-%d %H%M')))
    return labels

def multiDBSCAN(p, dist, t, plot=False, multiplier=1.0):
    params = []
    for eps in utils.epsRange():
        params.append(eps * multiplier)

    # DBSCAN is parallelized. This only speeds up the plotting
    assert len(params) > 0
    ret = Parallel(n_jobs=conf.nJobs if plot and conf.plotFile and len(params) > 1 else 1)(
        delayed(cluster)(p, dist, t, eps, plot) for eps in params
    )

    return {k: v for k, v in zip(params, ret)}

if __name__ == '__main__':
    with PhaseTimer('total'):
        s = loader.load()

        if conf.tstart is not None and conf.tend is not None:
            if conf.wnd is not None:
                # sliding window clustering
                tend = len(s.index) - conf.wnd + 1
                wndSize = conf.wnd
            else:
                # cluster entire loaded data
                tend = 1
                wndSize = len(s.index)
        else:
            # just cluster this single window
            tend = 1
            wndSize = conf.wnd

        assert tend >= 0, 'Loaded time shorter than window size'
        for t in tqdm(range(0, tend, conf.tstep)):
            p = s.iloc[slice(t, t + wndSize), :]
            print(t, p.index[0], '-', p.index[-1], ',', p.shape[0])
            assert p.shape[0] == wndSize

            dist = distance.dist(p)

            #print('clustering')
            multiDBSCAN(p, dist, p.index[0], True)


        if not conf.plotFile:
            plt.show()