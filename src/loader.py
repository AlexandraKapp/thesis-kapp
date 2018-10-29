# @Patrick Siegler

import pandas as pd
import numpy as np

import conf
from utils import tqdm


#loads data from h5 file into memory
def load():
    print('loading h5 store')
    store = pd.HDFStore(conf.wd + conf.data + '.h5', mode='r')
    full = store['df']

    print('sampling')

    #either use t (timestep) and wnd (window) or tstart and tend
    if conf.t is not None and conf.wnd is not None:
        t = np.searchsorted(full.index, pd.Timestamp(conf.t))
        tstart = full.index[t - 1]
        tend = full.index[t + conf.wnd - 1]
    else:
        tstart = pd.Timestamp(conf.tstart)
        tend = pd.Timestamp(conf.tend)

    print('Loading range', tstart, '-', tend)

    
    # assumes 0-indexed
    if conf.gridXmin == 0 and conf.gridXmax == conf.gridXdim - 1 and conf.gridYmin == 0 and conf.gridYmax == conf.gridYdim - 1:
        used_data_slice = full.loc[slice(tstart, tend), :]
    else:
        used_data_slice = pd.DataFrame()
        
        # tqdm for progress bar
        #only if gridXmin and gridXmax are defined. Otherwise full dataset is used
        for i in tqdm(range(conf.gridXmin, conf.gridXmax + 1)):
            x = full.loc[slice(tstart, tend), slice(i * conf.gridYdim + conf.gridYmin, i * conf.gridYdim + conf.gridYmax)]
            used_data_slice = used_data_slice.merge(x, how='outer', left_index=True, right_index=True, copy=False)

    print('sample size: {} = {}%'.format(used_data_slice.shape[0] * used_data_slice.shape[1], (used_data_slice.shape[0] * used_data_slice.shape[1] / (full.shape[0] * full.shape[1])) * 100))
    store.close()

    if conf.diff:
        #Calculates the difference of a DataFrame element compared with another element in the DataFrame (default is the element in the same column of the previous row).
        used_data_slice = used_data_slice.diff().shift(-1)
    #hack: drop those snp tickers that screw up the scale
    #p.drop(p.columns[np.where((p > 200).any())[0]], axis=1, inplace=True)

    used_data_slice.fillna(0, inplace=True)
    return used_data_slice
