from libc.math cimport log, exp
import numpy as np
cimport numpy as np



def entropyScore(np.ndarray[np.float64_t, ndim=2] dist, np.ndarray[np.int64_t] indices):
    if dist is None:
        return None

    # E(X) = - S_j( 1/m log( S_i( exp(-d(x_i, x_j, t ) 1/m ) )
    #      = - 1/m S_j( log( 1/m S_i( exp(-d(x_i, x_j, t ) ) )
    s1 = 0
    for j in indices:
        s2 = 0
        for i in indices:
            s2 = s2 + exp(-dist[i,j])
        s2 = s2 / len(indices)
        s1 = s1 + log(s2)
    s1 = s1 / len(indices)

    return -s1


def entropyScoreDistIndex(dist, np.ndarray[np.int64_t] indices):
    if dist is None:
        return None

    # E(X) = - S_j( 1/m log( S_i( exp(-d(x_i, x_j, t ) 1/m ) )
    #      = - 1/m S_j( log( 1/m S_i( exp(-d(x_i, x_j, t ) ) )
    s1 = 0
    for j in indices:
        s2 = 0
        for i in indices:
            s2 = s2 + exp(-dist.get(i,j))
        s2 = s2 / len(indices)
        s1 = s1 + log(s2)
    s1 = s1 / len(indices)

    return -s1
