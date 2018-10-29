#@Patrick Siegler

import numpy as np

from performance import PhaseTimer
from utils import tqdm
import BlockCorr
import conf

def corrnorm(s):
    tend = len(s.index) - conf.wnd + 1

    assert tend >= 0, 'Loaded time shorter than window size'
    assert conf.corr

    start = 0
    ti = 0
    preRho = None
    score = np.empty(int((len(s) - conf.wnd) / conf.tstep + 1))
    for t in tqdm(range(start, tend, conf.tstep)):
        p = s.iloc[slice(t, t + conf.wnd), :]
        if conf.verbose:
            print(t, p.index[0], '-', p.index[-1], ',', p.shape[0])
        assert p.shape[0] == conf.wnd

        with PhaseTimer('corrnorm'):
            #rho = BlockCorr.Pearson(p.T, conf.nJobs)
            rho = np.corrcoef(p, rowvar=0)
            rho = np.nan_to_num(rho)
            if ti > 0:
                diff = rho - preRho
                score[ti-1] = np.linalg.norm(diff)

            preRho = rho

        ti += 1
    score[ti-1] = np.nan

    return score