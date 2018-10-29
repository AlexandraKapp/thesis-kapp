import numpy as np
cimport numpy as np


def avgDist(nClusters_pre, np.ndarray[np.float64_t, ndim=2] dist):
    distances = np.zeros([nClusters_pre, nClusters_pre])
    for i in range(nClusters_pre):
        for j in range(i + 1, nClusters_pre):
            print ('i ', i)
            print ('j ', j)
            d = dist[i * 3 + 0, j * 3 + 0]
            d += dist[i * 3 + 0, j * 3 + 1]
            d += dist[i * 3 + 0, j * 3 + 2]
            d += dist[i * 3 + 1, j * 3 + 0]
            d += dist[i * 3 + 1, j * 3 + 1]
            d += dist[i * 3 + 1, j * 3 + 2]
            d += dist[i * 3 + 2, j * 3 + 0]
            d += dist[i * 3 + 2, j * 3 + 1]
            d += dist[i * 3 + 2, j * 3 + 2]
            d /= 9
            distances[i,j] = d
            distances[j,i] = d
    return distances

def avgDistGiven(nClusters_pre, np.ndarray[np.float64_t, ndim=2] dist, np.ndarray[np.int32_t] reps):

    cdef np.ndarray[np.float64_t, ndim=2] distances = np.zeros([nClusters_pre, nClusters_pre])
    cdef int i1, i2, i3, j1, j2, j3

    for i in range(nClusters_pre):
        i1 = reps[i * 3 + 0]
        i2 = reps[i * 3 + 1]
        i3 = reps[i * 3 + 2]
        for j in range(i + 1, nClusters_pre):
            j1 = reps[j * 3 + 0]
            j2 = reps[j * 3 + 1]
            j3 = reps[j * 3 + 2]

            d = dist[i1, j1]
            d += dist[i1, j2]
            d += dist[i1, j3]
            d += dist[i2, j1]
            d += dist[i2, j2]
            d += dist[i2, j3]
            d += dist[i3, j1]
            d += dist[i3, j2]
            d += dist[i3, j3]
            d /= 9
            distances[i,j] = d
            distances[j,i] = d
    return distances