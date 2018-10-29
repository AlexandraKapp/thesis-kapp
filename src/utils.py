# @Patrick Siegler

import numpy as np

import conf
import os

if conf.progress:
    import tqdm as _tqdm

try:
    RUNID = os.environ['LSB_JOBID']
except KeyError:
    RUNID = -np.random.randint(1, 999999)


def epsRange():
    if conf.clustering == 'Spectral':
        return [0]

    range = []
    eps = conf.eps
    if conf.epsDelta is None:
        range.append(eps)
    else:
        while (eps >= conf.epsLower):
            range.append(eps)
            eps = round(eps - conf.epsDelta, 5)

    return range

def makedirs(name):
    dirname = os.path.dirname(name)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)

# Sort a Correlation/Covariance matrix by cluster labels
def sortMatByLabels(m, labels):
    j = len(labels) - 1
    l = 0
    for i in range(len(labels)):
        while (labels[i] != l) and l <= max(labels):
            while labels[j] != l and j > i:
                j -= 1
            if j <= i:
                j = len(labels) - 1
                l += 1
            else:
                x = labels[i]
                labels[i] = labels[j]
                labels[j] = x

                x = m[i, :].copy()
                m[i, :] = m[j, :]
                m[j, :] = x

                x = m[:, i].copy()
                m[:, i] = m[:, j]
                m[:, j] = x

    return m

# Covert a Covariance Matrix to a Correlation Matrix
def CovToCor(A):
    D = np.diag(np.sqrt(np.diag(A)))
    D = np.linalg.inv(D)
    return np.dot(D, np.dot(A, D))

def tqdm(iterable):
    if conf.progress:
        return _tqdm.tqdm(iterable)
    else:
        return iterable