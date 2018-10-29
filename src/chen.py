#@Patrick Siegler

from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import itertools
from numpy.ma import exp, log
import numpy as np
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed

import conf
import distance
import loader
import cluster
from performance import PhaseTimer
from utils import tqdm

CYTHON = True

if CYTHON:
    import pyximport
    pyximport.install(setup_args={'include_dirs': np.get_include()})
    import cchen

# set to 1 for chen-like windows {t-b, t-b+1, ... t}{t, t+1, ... t+b}{t+b, t+b+1, ... t+2b} with overlap at t
# 0 for aligned windows of size b
WNDX = 1

def entropyScore(dist, labels, label, cache=None):
    indices = np.where(labels == label)[0]

    if cache is not None:
        tindices = tuple(indices)
        if tindices in cache:
            return cache[tindices]

    if CYTHON:
        if conf.clustering == 'FastDBSCAN':
            result = cchen.entropyScoreDistIndex(dist, indices)
        else:
            result = cchen.entropyScore(dist, indices)
    else:
        if dist is None:
            return None


        # E(X) = - S_j( 1/m log( S_i( exp(-d(x_i, x_j, t ) 1/m ) )
        #      = - 1/m S_j( log( 1/m S_i( exp(-d(x_i, x_j, t ) ) )
        s1 = 0
        for j in indices:
            s2 = 0
            for i in indices:
                if conf.clustering == 'FastDBSCAN':
                    s2 = s2 + exp(-dist.get(i,j))
                else:
                    s2 = s2 + exp(-dist[i,j])
            s2 = s2 / len(indices)
            s1 = s1 + log(s2)
        s1 = s1 /len(indices)

        result = -s1

    if cache is not None:
        cache[tindices] = result
    return result

def chenTest(ctx, eps, i):
    labels = ctx.clusters[eps]

    score = entropyScore(ctx.dist, labels, i, ctx.cache)
    if score == 0:
        return (None, None)

    if ctx.distPre is not None:
        preScore = entropyScore(ctx.distPre, labels, i, ctx.cachePre)
        preFrac = preScore / score
    else:
        (preScore, preFrac) = (None, None)
    if ctx.distPost is not None:
        postScore = entropyScore(ctx.distPost, labels, i, ctx.cachePost)
        postFrac = postScore / score
    else:
        (postScore, postFrac) = (None, None)

    if not preFrac is None and preFrac > 200:
        preFrac = None
    if not postFrac is None and postFrac > 200:
        postFrac = None

    if conf.verbose and False:
        print(eps, i, preScore, score, postScore, end=' ')
        if preFrac is not None and preFrac > conf.chenTh:
            print('\033[1m', preFrac, '\033[0m', end=' ')
        else:
            print(preFrac, end=' ')
        if postFrac is not None and postFrac > conf.chenTh:
            print('\033[1m', postFrac, '\033[0m')
        else:
            print(postFrac)


    return (preFrac, postFrac)


def runChenTest(ctx, params):
    ret = []
    for (eps, i) in params:
        ret.append(chenTest(ctx, eps, i))

    return ret


def chen(s):
    tend = len(s.index) - conf.wnd - WNDX

    scores = []
    assert len(s.index) - 2 * conf.wnd - WNDX >= 0, 'Loaded time shorter than double window size'
    distPre = None
    dist = None
    distPost = None
    cachePre = {}
    cache = {}
    cachePost = {}
    for t in tqdm(range(0, tend, conf.tstep)):
        if (t + conf.wnd) > tend:
            break
        p = s.iloc[slice(t, t + conf.wnd + WNDX), :]
        if conf.verbose:
            print(t, p.index[0], '-', p.index[-1], ',', p.shape[0])

        with PhaseTimer('chen distance'):
            if dist is None:
                assert t == 0  # data start, later this is set from distPost
                dist = distance.dist(p)

            if distPre is None:
                assert t < conf.wnd  # data start

            pPost = s.iloc[slice(t + conf.wnd, t + 2 * conf.wnd + WNDX), :]
            if (pPost.shape[0] == conf.wnd + WNDX):
                distPost = distance.dist(pPost)
            else:
                assert t >= len(s.index) - 2 * conf.wnd  # data end
                distPost = None

            if distPre is None and distPost is None:
                continue

        with PhaseTimer('chen clustering'):
            clusters = cluster.multiDBSCAN(p, dist, t)

        params = []
        for eps in clusters.keys():
            labels = clusters[eps]
            nClusters = max(labels) + 1
            for i in range(nClusters):
                params.append((eps, i))
        jobParams = np.array_split(params, 1 if conf.chenNoHits else conf.nJobs)

        with PhaseTimer('chen test'):
            ctx = SimpleNamespace(s=s, p=p, clusters=clusters, pPost=pPost, distPre=distPre, dist=dist,
                                  distPost=distPost, t=t, cachePre=cachePre, cache=cache, cachePost=cachePost)

            ret = Parallel(n_jobs=1 if conf.chenNoHits else conf.nJobs)(  # TODO: speedup only if many series or many hits (=plotting)
                delayed(runChenTest)(ctx, params) for params in jobParams
            )

            df = pd.DataFrame(list(itertools.chain.from_iterable(ret)))
            if len(df) > 0:
                scores.append((np.max(df[0]), np.mean(df[0]), np.sum(df[0]), np.max(df[1]), np.mean(df[1]), np.sum(df[1])))
            else:
                scores.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))

        distPre = dist
        dist = distPost
        distPost = None
        cachePre = cache
        cache = cachePost
        cachePost = {}

    sdf = pd.DataFrame(scores, columns=['preMax', 'preMean', 'preSum', 'postMax', 'postMean', 'postSum'])
    sdf.preMax = sdf.preMax.shift(-1)
    sdf.preMean = sdf.preMean.shift(-1)
    sdf.preSum = sdf.preSum.shift(-1)
    sdf['ChenMax'] = sdf.preMax + sdf.postMax
    sdf['ChenMean'] = sdf.preMean + sdf.postMean
    sdf['ChenSum'] = sdf.preSum + sdf.postSum
    sdf.drop(['preMax', 'preMean', 'preSum', 'postMax', 'postMean', 'postSum'], axis=1, inplace=True)
    return sdf

if __name__ == '__main__':
    if conf.clustering != "DBSCAN" and conf.clustering != "FastDBSCAN":
        conf.parser.error("chen requires MultiDBSCAN (DBSCAN or FastDBSCAN)")
    if conf.epsDelta is None:
        conf.parser.error("chen requires epsDelta and epsLower")
    if conf.tstart is None or conf.tend is None:
        conf.parser.error("chen requires tstart and tend")
    if conf.wnd is None or conf.t is not None:
        conf.parser.error("chen requires wnd and no t")
    if conf.chenTh is None:
        conf.parser.error("chen requires chenTh")

    with PhaseTimer('total'):

        s = loader.load()

        chen(s)

    if not conf.plotFile:
        plt.show()