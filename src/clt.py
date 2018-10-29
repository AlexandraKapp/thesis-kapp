import numpy as np
import pandas as pd
from utils import tqdm
import pickle
import os
import errno

import conf
if conf.plotFile:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cluster
import utils
import loader
import distance
from performance import PhaseTimer
import clt_stats


import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})


L2TRANSITIONS=True

class TraceGraphLayer:
    """A layer includes all relevant cluster information for a time window.
    """


    def __init__(self, labels,data_slice):
        # array: cluster label for each time series
        self.labels = labels

        # array: ext windows cluster label for each time series
        self.next_labels = None

        # array: counts of time series per cluster
        self.counts = None

        # array: id of cluster where the mainstream of this cluster follows
        self.mainstreamId = None

        # matrix(shape=count of previous clusters, count of current clusters):
        # the count of transition from time series that wandered from pre to current in the respective cell
        self.transitions = None

        # DataFrame: data of current layer
        self.data_slice = data_slice

        # DataFrame: data of next layer
        self.next_data_slice = None


def cluster_and_trace(data):
    """Data is clustered for each window and cluster traces are computed.
       Args:
           data (DataFrame): matrix (n x t) with n time series for t time steps

       Returns:
           dict: A dictionary with eps as keys and TraceGraphLayer as value
       """

    tend = len(data.index) - conf.wnd + 1

    assert tend >= 0, 'Loaded time shorter than window size'
    traces = {}
    for eps in utils.epsRange():
        traces[eps] = []

    time_step_index = 0
    start = 0
    dist = None

    # do for each time window cluster and tracing
    for time_step_date in tqdm(range(start, tend, conf.tstep)):
        current_data_slice = data.iloc[slice(time_step_date, time_step_date + conf.wnd), :]
        if conf.verbose:
            print(time_step_index, time_step_date, current_data_slice.index[0], '-', current_data_slice.index[-1], ',', current_data_slice.shape[0])
        assert current_data_slice.shape[0] == conf.wnd

        with PhaseTimer('clt distance'):
            #get distance between time series to calc multiDBSCAN
            #distance only needed for DBSCAN and corrnorm
            dist = distance.dist(current_data_slice)

        with PhaseTimer('clt clustering'):
            # cluster data
            clusters = cluster.multiDBSCAN(current_data_slice, dist, current_data_slice.index[0])

        with PhaseTimer('clt tracing'):
            for eps in clusters.keys():
                labels = clusters[eps]
                trace = traces[eps]

                trace.append(TraceGraphLayer(labels, current_data_slice))

                if (time_step_index > 0):
                    # get time series labels from previous time step
                    labels_pre = trace[time_step_index - 1].labels
                    # amount of clusters
                    n_clusters = max(labels) + 1
                    nClusters_pre = max(labels_pre) + 1

                    counts = np.zeros(nClusters_pre)
                    transitions = np.zeros([nClusters_pre, n_clusters])
                    representatives = []
                    for _ in range(nClusters_pre):
                        representatives.append([])

                    for i in range(len(labels_pre)):
                        if labels_pre[i] != -1:
                            counts[labels_pre[i]] += 1

                            #representatives are reps of a cluster indicating the index of the time series
                            representatives[labels_pre[i]].append(i)
                            if labels[i] != -1:
                                transitions[labels_pre[i]][labels[i]] += 1

                    trace[time_step_index - 1].counts = counts
                    trace[time_step_index - 1].transitions = transitions
                    trace[time_step_index - 1].nextLabels = labels
                    trace[time_step_index - 1].next_data_slice = current_data_slice

                    if time_step_index > 1:
                        trace[time_step_index -2].next2_data_slice = current_data_slice
                    if time_step_index > 2:
                        trace[time_step_index-3].next3_data_slice = current_data_slice


                #needed for OffstreamRatio 2x
                if L2TRANSITIONS and (time_step_index > 1):
                    labels_pre2 = trace[time_step_index - 2].labels
                    n_clusters = max(labels) + 1
                    n_clusters_pre2 = max(labels_pre2) + 1

                    transitions2 = np.zeros([n_clusters_pre2, n_clusters])
                    for i in range(len(labels_pre2)):
                        if labels_pre2[i] != -1:
                            if labels[i] != -1:
                                transitions2[labels_pre2[i]][labels[i]] += 1

                    trace[time_step_index - 2].transitions2 = transitions2

        time_step_index += 1
    return traces


def trace_or_load():
    """load traces they are in cache, otherwise compute
    """
    tend = len(data.index) - conf.wnd + 1
    pklName = 'clt/cache/trace_{}_grid{}{}{}{}_w{}_ts{}_eps{:.3f}_{}_{}_{}_{}.pkl'\
        .format(conf.data, conf.gridXmin, conf.gridXmax, conf.gridYmin, conf.gridYmax, conf.wnd,
                conf.tstep, conf.eps, conf.epsDelta, conf.epsLower,
                data.index[0].strftime('%Y-%m-%d_%H%M'), data.index[tend].strftime('%Y-%m-%d_%H%M'))

    if False and os.path.exists(pklName) and conf.clustering == 'DBSCAN' and conf.data in ['mobile', 'evi', 'snp']:
        print('\033[91m=============================================================================\033[0m')
        print('\033[91m\033[1mWARN: Using previously saved graph. Non-persisted options might have changed!')
        print('\033[91m=============================================================================\033[0m')
        with open(pklName, 'rb') as f:
            traces = pickle.load(f)
    else:
        traces = cluster_and_trace(data)
    return traces


def shadowed_chen(data):
    cl = conf.clustering
    e = conf.eps
    el = conf.epsLower
    ed = conf.epsDelta

    conf.clustering = 'DBSCAN'
    conf.eps = conf.cltChenEps
    conf.epsDelta = conf.cltChenEpsDelta
    conf.epsLower = conf.cltChenEpsLower

    import chen
    out = chen.chen(data)

    conf.clustering = cl
    conf.eps = e
    conf.epsLower = el
    conf.epsDelta = ed

    return out


if __name__ == '__main__':
    if conf.tstart is None or conf.tend is None:
        conf.parser.error("clt requires tstart and tend")
    if conf.wnd is None or conf.t is not None:
        conf.parser.error("clt requires wnd and no t")

    with PhaseTimer('total'):
        with PhaseTimer('load'):
            data = loader.load()

        print('Cluster&Trace...')
        traces = trace_or_load()

        # compat for statistics using L2 stats - they can't use the simplified version because it doesn't keep L2 consistent
        for eps in traces.keys():
            trace = traces[eps]
            for ti in range(0, len(trace)):
                trace[ti].origcounts = np.copy(trace[ti].counts)
                trace[ti].origtransitions = np.copy(trace[ti].transitions)

        nonTraceStats = pd.DataFrame(index=np.arange(0, len(trace)))

        if conf.cltIncludeChen:
            print('Run Chen...')
            nonTraceStats = pd.merge(nonTraceStats, shadowed_chen(data), right_index=True, left_index=True, how='left')

        if conf.cltIncludeNorm:
            import corrnorm
            print('Run Norm...')
            nonTraceStats['corrnorm'] = corrnorm.corrnorm(data)

        print('Measure...')
        with PhaseTimer('clt measure'):
            store = pd.HDFStore(conf.wd + conf.data  + '.h5', mode='r')
            change_points = store['change_points'] if '/change_points' in store.keys() else None
            store.close()

            allstats = clt_stats.measure(traces)

            for eps in traces.keys():
                allstats[eps] = pd.merge(allstats[eps], nonTraceStats, right_index=True, left_index=True)



            #write score values file
            filename = conf.data.split('\\')
            filename = filename[len(filename) - 1].split('/')
            filename = filename[len(filename) - 1]

            #store score values
            if conf.storeScores:
                f = conf.evaluationDir + '\\scores\\scores' + '_{}_{}_w{}_ts{}_eps{:.5f}_{}_{}.csv'. \
                    format(filename, conf.clustering, conf.wnd, conf.tstep, eps,
                            conf.tstart, conf.tend)

                if not os.path.exists(os.path.dirname(f)):
                    try:
                        os.makedirs(os.path.dirname(f))
                    except OSError as exc:  # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise

                with open(f, 'a+', newline='\n') as csv_file:
                    for key, value in sorted(allstats.items()):
                        csv_file.write(value.to_csv())

            if change_points is not None:
                clt_stats.score(allstats, change_points)

        if not conf.cltNoStats:
            with PhaseTimer('clt plot stats'):
                clt_stats.plot(data, allstats, change_points)

    if not conf.plotFile:
        plt.show()