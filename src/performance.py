# @Patrick Siegler

import timeit
import datetime
import atexit

import conf
import utils

_times = {}

class PhaseTimer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.tstart = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = timeit.default_timer() - self.tstart
        if self.name in _times:
            _times[self.name] += elapsed
        else:
            _times[self.name] = elapsed

def _atexit():

    with open(conf.evaluationDir + '\\perf.log', 'a') as log:
        for name in _times.keys():
            print('Time spent in', name, 'is', str(datetime.timedelta(seconds=_times[name])))
            print(utils.RUNID, conf.gridXdim*conf.gridYdim, conf.Nsample, conf.infFrac, conf.infNoise, conf.infChangeFrac, conf.Nevt,
                      conf.wnd, conf.tstep, name.replace(' ', '_'), _times[name], file=log)

atexit.register(_atexit)