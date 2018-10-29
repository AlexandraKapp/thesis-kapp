#@Patrick Siegler

import numpy as np
import pandas as pd
import conf
from sklearn.metrics.pairwise import pairwise_distances
from performance import PhaseTimer
import BlockCorr
from math import sqrt
import atexit
import utils

CYTHON = True
distances = 0


def dist(series, force=False):
    #if conf.clustering == 'BlockCorr' and not force:
    #    #todo distance counting in BlockCorr
    #    #return None
    #    return distIndex(series)
    if conf.clustering == 'FastDBSCAN' and not force:
         return distIndex(series)

    if conf.corr:
        global distances
        distances += series.shape[1] * (series.shape[1] - 1) / 2

        dist = np.corrcoef(series, rowvar=0) #series.corr()
        dist = np.nan_to_num(dist)
        dist = np.sqrt(1 - np.square(dist)) #.abs()

    else:
        dist = pairwise_distances(series.transpose(), n_jobs=conf.nJobs)

    return dist


class distIndex(object):
    def __init__(self, series):
        global distances
        assert conf.corr, "Index is implemented for correlation only ATM"
        self.series = np.asarray(series.T, order='C')

        self.cache = {}
        self.pivot = 0
        self.index = pd.Series(np.empty(series.shape[0]))
        for i in range(series.shape[0]):
            distances += 1
            self.index[i] = sqrt(1 - BlockCorr.Pearson2(self.series, self.pivot, i)**2)
        self.index.sort_values(inplace=True)

    def get(self, i, j):
        global distances
        if i == self.pivot:
            return self.index.loc[j]
        if j == self.pivot:
            return self.index.loc[i]

        if (i,j) in self.cache:
            return self.cache[(i,j)]
        elif (j,i) in self.cache:
            return self.cache[(j,i)]
        else:
            distances += 1
            self.cache[(i,j)] = sqrt(1 - BlockCorr.Pearson2(self.series, i, j)**2)
            return self.cache[(i,j)]

    # optimized version does not include i in result !
    def region_query(self, i, eps):
        with PhaseTimer('regionQuery'):
            iv = self.index.values
            iiv = self.index.index.values

            k = self.index.index.get_loc(i)
            kr = self.index.values[k]
            out = []

            j = k - 1
            while j >= 0 and kr - iv[j] < eps:
                d = self.get(i, iiv[j])
                if d < eps:
                    out.append(iiv[j])
                j -= 1
            min = j

            j = k + 1
            upper = self.series.shape[0]
            while j < upper and iv[j] - kr < eps:
                d = self.get(i, iiv[j])
                if d < eps:
                    out.append(iiv[j])
                j += 1
            max = j

            return out

    def size(self):
        return self.series.shape[0]

