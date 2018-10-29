import numpy as np
cimport numpy as np
import pandas as pd
import conf
from sklearn.metrics.pairwise import pairwise_distances
from performance import PhaseTimer
import BlockCorr
from math import sqrt
from libcpp.map cimport map
from cython.operator cimport dereference
import distance

PRECOMPUTE = False
CACHE_MATRIX = True #True for np array, false for std::map


cdef class distIndex(object):
    cdef object series
    cdef object index
    cdef object cacheM
    cdef map[int,float] cache
    cdef int pivot
    cdef int _size
    cdef object dist

    def __init__(self, series):
        assert conf.corr, "Index is implemented for correlation only ATM"
        self.pivot = 0
        self.index = pd.Series(np.empty(self._size))

        if PRECOMPUTE:
            distance.distances += series.shape[1] * (series.shape[1] - 1) / 2
            dist = BlockCorr.Pearson(series.T) #series.corr()
            self.dist = np.sqrt(1 - np.square(dist)) #.abs()
            self._size = self.dist.shape[0]

            for i in range(self._size):
                self.index[i] = self.get(self.pivot, i)
        else:
            self.series = np.asarray(series.T, order='C')
            self._size = self.series.shape[0]

            if CACHE_MATRIX:
                self.cacheM = np.empty((self._size, self._size))
                self.cacheM.fill(-1)
            else:
                self.cache = map[int,float]()
            for i in range(self._size):
                distance.distances += 1
                self.index[i] = sqrt(1 - BlockCorr.Pearson2(self.series, self.pivot, i)**2)

        self.index.sort_values(inplace=True)

    def get(self, int i, int j):
        if PRECOMPUTE:
            return self.dist[i,j]

        if i == self.pivot:
            return self.index.loc[j]
        if j == self.pivot:
            return self.index.loc[i]

        if CACHE_MATRIX:
            if self.cacheM[i,j] != -1:
                return self.cacheM[i,j]
            else:
                distance.distances += 1
                p = sqrt(1 - BlockCorr.Pearson2(self.series, i, j)**2)
                self.cacheM[i,j] = p
                self.cacheM[j,i] = p
                return self.cacheM[i,j]
        else:
            it = self.cache.find(i * self._size + j)
            if it != self.cache.end():
                return dereference(it).second
            else:
                it = self.cache.find(j * self._size + i)
                if it != self.cache.end():
                    return dereference(it).second
                else:
                    distance.distances += 1
                    self.cache[i * self._size + j] = sqrt(1 - BlockCorr.Pearson2(self.series, i, j)**2)
                    return self.cache[i * self._size + j]

    # optimized version does not include i in result !
    def region_query(self, i, eps):
        cdef np.ndarray[np.float64_t] iv = self.index.values
        cdef np.ndarray[np.int64_t] iiv = self.index.index.values

        k = self.index.index.get_loc(i)
        kr = self.index.values[k]
        out = []

        j = k - 1
        while j >= 0 and kr - iv[j] < eps:
            if self.get(i, iiv[j]) < eps:
                out.append(iiv[j])
            j -= 1
        min = j

        j = k + 1
        while j < self._size and iv[j] - kr < eps:
            if self.get(i, iiv[j]) < eps:
                out.append(iiv[j])
            j += 1
        max = j

        #print ((max - min) * 1.0 / self._size)
        return out

    def size(self):
        return self._size
