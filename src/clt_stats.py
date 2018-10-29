import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import csv
import os
import errno
import glob

import conf
import plotting
import evaluation


columns = [ 'Ncluster', 'Noffstream','Weighted Noffstream', 'OffstreamRatio', 'OffstreamRatio2X', 'Weighted OffstreamRatio',
            'Inter-Cluster Distance','Weighted Inter-Cluster Distance', 'Inter-Cluster-Change', 'Weighted Inter-Cluster-Change', 'Survivor-Distance-Change', 'Weighted Survivor-Distance-Change']

filename = conf.data.split('\\')
filename = filename[len(filename) - 1].split('/')
filename = filename[len(filename) - 1]


def measure(traces, SURVIVAL_THRESHOLD = 0.8):
    '''Compute change detection scores for the given layer.

    Attributes:
    traces (dictionary of TraceGraphLayers): cluster traces for each given eps
    SURVIVAL_THRESHOLD (float, default: 0.8): defines how many time series
            need to survive within one cluster so the cluster is considered a survivor cluster
    '''

    allstats = {}

    for eps in traces.keys():

        trace = traces[eps]
        stats = pd.DataFrame(
            columns=columns, index=np.arange(0, len(trace)))

        OffstreamRatio2prev = np.nan

        for ti in range(0, len(trace) - 1):
            layer = trace[ti]

            # create a file that records the assignments of time series to clusters
            if conf.storeClusterAssign:
                f = conf.evaluationDir + '\\cluster_assignment\\assign_'  + '_{}_{}_w{}_ts{}_eps{:.5f}_{}_{}.csv'.\
                    format(filename, conf.clustering, conf.wnd, conf.tstep,eps, conf.tstart, conf.tend)

                if not os.path.exists(os.path.dirname(f)):
                    try:
                        os.makedirs(os.path.dirname(f))
                    except OSError as exc:  # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise

                with open(f,'a+',newline='\n') as file:
                    writer = csv.writer(file)
                    if os.stat( f).st_size == 0:
                        header = (np.arange(0,len(layer.labels)))
                        writer.writerow(header)

                    writer.writerow(layer.labels)

            Ncluster = len(layer.counts)
            Noffstream = 0
            OffstreamRatios = []
            OffstreamRatios2 = []

            Noffstream_weighted = 0
            OffstreamRatios_weighted = []
            Inter_cluster_distances = []
            Inter_cluster_distances_weighted = []
            Inter_cluster_changes = []
            Inter_cluster_changes_weighted = []
            Inter_cluster_changes_survivors = []
            Inter_cluster_changes_survivors_weighted = []


            # correlate all time series
            ts_corr_matrix = np.corrcoef(layer.data_slice, rowvar=0)
            ts_corr_matrix_next =np.corrcoef(layer.next_data_slice, rowvar=0)

            # get mainstream cluster for each cluster
            if layer.transitions.size > 0:
                mainstream = np.argmax(layer.transitions, axis=1)

            #get info for each cluster if it is a survivor
            is_cluster_survivor = []
            for i in range(0, len(layer.counts)):
                cluster_survivor_forward = np.where(layer.transitions[i] > layer.counts[i] * SURVIVAL_THRESHOLD)[0]
                cluster_survivor_backward = np.where(layer.counts[i] > (np.sum(layer.transitions[:, mainstream[i]]) * SURVIVAL_THRESHOLD))[0]
                is_cluster_survivor.append((len(cluster_survivor_forward) > 0) and (len(cluster_survivor_backward) > 0))

            NseriesSurvivor = 0

            #iteration over cluster
            for cl in range(len(layer.counts)):

                #get offstream scorces
                if len(layer.transitions[cl]) > 0:

                    # ------START OFFSTREAM CALC---------

                    Noffstream += (layer.counts[cl] - np.max(layer.transitions[cl]))
                    OffstreamRatios.append(1 - (np.max(layer.transitions[cl]) / layer.counts[cl]))
                    ts_not_in_mainstream = np.argwhere(layer.nextLabels != mainstream[cl])

                    #locate offstream time series
                    offstream_ts = ts_not_in_mainstream[np.isin(ts_not_in_mainstream,  np.argwhere(layer.labels == cl))]

                    if is_cluster_survivor[cl]:
                        NseriesSurvivor += (len(np.where(layer.labels == cl)[0]) - len(offstream_ts))

                    single_cluster_offstream = 0
                    if len(offstream_ts) > 0:
                        # get all time series that are part of mainstream cluster
                        ts_in_mainstream = np.argwhere(layer.nextLabels == mainstream[cl]).flatten()

                        for x in offstream_ts:
                            #calc weighted mean distance to mainstream cluster
                            #the closer (highly correlated) the smaller the value

                            #offstream_mean_dist = np.sqrt(1 - np.square(ts_corr_matrix_next[x, ts_in_mainstream]).mean())
                            #offstream_mean_dist = np.sqrt(1 - np.square(ts_corr_matrix_next[x, ts_in_mainstream])).mean()
                            offstream_mean_dist = 1 - abs(ts_corr_matrix_next[x, ts_in_mainstream]).mean()

                            single_cluster_offstream += offstream_mean_dist
                            Noffstream_weighted += offstream_mean_dist

                            #get ratio of offstream time series in comparison to total ts count of cluster
                    OffstreamRatios_weighted.append(single_cluster_offstream / layer.counts[cl])

                    #-------END OFFSTREAM CALC------------



                    if cl < len(layer.counts) - 1:

                        #------START INTERCLUSTER CALC---------

                        # get all time series of current cluster in next time step
                        ts_cl_next = np.argwhere(layer.nextLabels == mainstream[cl])

                        # iterate over all other clusters and get each mean correlation
                        mean_dist_list = list(range(len(layer.counts)-(cl+1)))
                        mean_dist_next_list = list(range(len(layer.counts)-(cl+1)))
                        p_next_counts = list(range(len(layer.counts)-(cl+1)))

                        for cl2 in range (cl+1, len(layer.counts)):

                            mean_dist_list[cl2 - (cl + 1)] = 1 - (ts_corr_matrix[np.argwhere
                            (layer.labels == cl2), np.argwhere(layer.labels == cl).flatten()]).mean()

                            #mean_dist_list [cl2-(cl+1)] = np.sqrt(1 - np.square(ts_corr_matrix[np.argwhere(layer.labels == cl2),
                            #                                                         np.argwhere(layer.labels == cl).flatten()])).mean()

                            cl2_next = np.argwhere(layer.nextLabels == mainstream[cl2])
                            p_next_counts [cl2-(cl+1)] = (len(cl2_next))

                            mean_dist_next_list [cl2-(cl+1)] = 1 - (ts_corr_matrix_next[cl2_next, ts_cl_next.flatten()]).mean()
                            #mean_dist_next_list[cl2 - (cl + 1)] = np.sqrt(1 - np.square(ts_corr_matrix_next[cl2_next,
                            #                                                            ts_cl_next.flatten()])).mean()

                        # calc values for mean distances between clusters
                        Inter_cluster_distances.append(np.mean(mean_dist_list))
                        weighted_mean_dist = mean_dist_list * (layer.counts[cl+1:] + layer.counts[cl])
                        Inter_cluster_distances_weighted.append(np.mean(weighted_mean_dist))

                        # calc values for mean diff between t and t+1 between clusters
                        mean_diff_list = abs(np.subtract(mean_dist_list, mean_dist_next_list))
                        mean_diff = np.mean(mean_diff_list)
                        weighted_mean_diff = np.multiply(mean_diff_list, np.add(p_next_counts, len(ts_cl_next)))
                        Inter_cluster_changes.append(mean_diff)
                        # weight by ts survior counts of each cluster
                        Inter_cluster_changes_weighted.append(np.mean(weighted_mean_diff))

                        if is_cluster_survivor[cl] and len(np.where(is_cluster_survivor[cl+1:])[0]) > 0:
                            diff_survivor = np.array(mean_diff_list)[is_cluster_survivor[cl+1:]]
                            cl2_counts = layer.counts[cl+1:][is_cluster_survivor[cl+1:]]
                            cl2_next_counts = np.array(p_next_counts)[is_cluster_survivor[cl+1:]]
                            weighted_mean_survivor = diff_survivor * (cl2_counts + layer.counts[cl])
                            Inter_cluster_changes_survivors.append(np.mean(diff_survivor))
                            Inter_cluster_changes_survivors_weighted.append(np.mean(weighted_mean_survivor))

                        # -------END INTERCLUSTER CALC------------


            Inter_cluster_distance = np.average(Inter_cluster_distances)
            Inter_cluster_distance_weighted = np.average(Inter_cluster_distances_weighted)
            Inter_cluster_change = np.average(Inter_cluster_changes)
            Inter_cluster_change_weighted = np.average(Inter_cluster_changes_weighted)
            Inter_cluster_change_survivors = np.average(Inter_cluster_changes_survivors)
            Inter_cluster_change_survivors_weighted = np.average(Inter_cluster_changes_survivors_weighted)


            if hasattr(layer, 'transitions2'):
                for cl in range(len(layer.origcounts)):
                    if len(layer.transitions2[cl]) > 0:
                        OffstreamRatios2.append(1 - (np.max(layer.transitions2[cl]) / layer.origcounts[cl]))

            OffstreamRatio_weighted = np.average(OffstreamRatios_weighted)
            OffstreamRatio = np.average(OffstreamRatios)
            OffstreamRatio2 = np.average(OffstreamRatios2) if len(OffstreamRatios2) > 0 else np.nan

            OR2X = OffstreamRatio2 * OffstreamRatio2prev

            row = [Ncluster, Noffstream, Noffstream_weighted, OffstreamRatio, OR2X,
                   OffstreamRatio_weighted, Inter_cluster_distance, Inter_cluster_distance_weighted,
                   Inter_cluster_change, Inter_cluster_change_weighted, Inter_cluster_change_survivors, Inter_cluster_change_survivors_weighted]

            stats.loc[ti] = row

            OffstreamRatio2prev = OffstreamRatio2

        allstats[eps] = stats

    return allstats


def score(allstats, change_points):
    """compute recall, precision and F1 score and write them into csv file
    Arguments:
        allstats(DataFrame):matrix (nxm) with n scores and score values for m time windows
        change_points(DataFrame):in each row a change point is listed with respective attributes
    """

    # if data is artificial data, then retrieve noise and distance change from filename
    if filename.startswith('noise'):
        split_file = filename.split('_dist')
        noise = split_file[0].split('noise')[1]
        dist_change = split_file[1].split('_')[0]
    else:
        noise = ''
        dist_change = ''

    for key, value in sorted(allstats.items()):
        evaluation.compute_recPrecF1_allTh(value, value.columns, change_points, filename, noise, conf.eps,
                                                  dist_change, conf.clustering, conf.wnd, conf.tstep)
        evaluation.compute_CoD_allTh(value, value.columns, filename, noise, conf.eps,
                                                  dist_change, conf.clustering, conf.wnd, conf.tstep)


def plot(s, allstats, change_points):
    """create plots and store them as pdf
    Arguments:
        allstats(DataFrame):matrix (nxm) with n scores and score values for m time windows
        change_points(DataFrame):in each row a change point is listed with respective attributes
    """
    tend = len(s.index) - conf.wnd + 1

    for eps in allstats.keys():
        stats = allstats[eps]

        print('plotting stats')
        if conf.cltPlotFullStats:
            for col in stats.columns:
                stats[col].plot(linewidth=2)
                # plt.legend(fontsize=8)
                plt.gcf().set_size_inches(10, 1.5)
                plt.axhspan(np.mean(stats[col]) - 1.5 * np.std(stats[col]), np.mean(stats[col]) + 1.5 * np.std(stats[col]),
                        color='lightgray')
                plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.95, hspace=0.4, wspace=0.4)
                if change_points is not None:
                    for x in change_points.as_matrix():
                        plt.axvline(x=(x[0] / conf.tstep), color='gray' if x[1] == 'A' else 'blue')

                plotting._finalize(
                    'clt/detail/stats_{}_{}_{}_w{}_ts{}_s{}_eps{:.5f}_{}_{}'.format(filename, col, conf.clustering,
                                                                                         conf.wnd, conf.tstep,
                                                                                         eps,
                                                                                         s.index[0].strftime(
                                                                                             '%Y-%m-%d_%H%M'),
                                                                                         s.index[tend].strftime(
                                                                                             '%Y-%m-%d_%H%M')))
        stats = stats.apply(
            lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)) if (np.max(x) - np.min(x)) != 0 else np.zeros(
                len(stats)))
        for i in range(len(stats.columns)):
            stats.iloc[:, i] = stats.iloc[:, i] + i
        f = stats.plot(linewidth=2)
        #plt.legend(fontsize=8)
        plt.gcf().set_size_inches(4*5, 3*5)
        if change_points is not None:
            for x in change_points.as_matrix():
                plt.axvline(x=(x[0] / conf.tstep), color='gray' if x[1] == 'A' else 'blue')
        for col in stats.columns:
            plt.axhspan(np.mean(stats[col]) - 1.5 * np.std(stats[col]), np.mean(stats[col]) + 1.5 * np.std(stats[col]),
                        color='lightgray')
        plotting._finalize(
            '_{}_{}_w{}_ts{}_eps{:.5f}_{}_{}'.format(filename, conf.clustering, conf.wnd, conf.tstep,
                                                                       eps,
                                                                       s.index[0].strftime('%Y-%m-%d_%H%M'),
                                                                       s.index[tend].strftime('%Y-%m-%d_%H%M')))