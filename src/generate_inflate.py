import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import minimize
from random import randint
import random

import conf

INITAL_CORR_COEF = 0.0

# generate for each cluster one dummy time series that the other cluster members use as model
def generate_corr_model(corr_coef, cluster_count, changed=None, prev_model=None):
    cov = np.full((cluster_count, cluster_count), corr_coef)
    np.fill_diagonal(cov, 1)
    mean = (np.zeros(cov.shape[0]))

    new_model =  np.random.multivariate_normal(mean, cov, conf.Nsample)

    # if bc of the change fraction only a certain amount of clusters are to be changed, then these are picked out and
    # combined with the old model
    if changed is not None:
        combined_model = prev_model
        combined_model[:,changed]=new_model[:prev_model.shape[0],changed]
        return combined_model

    return new_model

#assign model: probability a time series is assigned to a cluster
def assign(assign_model, p):
    if conf.infMethod == 'uniform':
        return np.random.randint(0, p)
    elif conf.infMethod == 'dirichlet':
        return np.random.multinomial(1, assign_model).argmax()
    else:
        raise NotImplementedError

#assigns: the assigned cluster of each time series (shape =(time series count,))
def assign_initial(cluster_count, series_count):
    if conf.infMethod == 'uniform':
        return (None, np.random.randint(0, cluster_count, series_count))
    #Sie ist die multivariate Erweiterung der Beta-Verteilung und die konjugierte A-priori-Verteilung der multinomialen Verteilung in Bayesscher Statistik
    elif conf.infMethod == 'dirichlet':
        assign_model = np.random.dirichlet(np.ones(cluster_count) * cluster_count) #TODO: p or a fix number? how does this change with p?
        assigns = np.random.multinomial(1, assign_model, series_count).argmax(1)
        return (assign_model, assigns)
    else:
        raise NotImplementedError


def generate_data(model, assigns, steps):
    #create noise with the standard deviation defined in conf.noise and a mean of 0
    dat = np.random.normal(0, conf.infNoise, (steps, series_count))
    for i in range(dat.shape[1]):
        model_series = model[:steps, assigns[i]]
        dat[:, i] += model_series
    return dat


def generate_model(corr_coef, prev_steps, prev_model, assign_model, assigns, type, change_frac):
    #generate initial model

    if prev_model is None:
        (assign_model, assigns) = assign_initial(cluster_count, series_count)
        src_model = generate_corr_model(corr_coef, cluster_count)

    else:
        if type == 'A':
            src_model = prev_model[prev_steps:, ]
            changed = np.random.choice(range(series_count), int(series_count * change_frac), replace=False)
            for i in changed:
                assigns[i] = assign(assign_model, cluster_count)
        elif type == 'B':

            #change as many clusters as according to change_frac
            changed = np.random.choice(range(cluster_count), int(cluster_count * change_frac), replace=False)

            src_model = generate_corr_model(corr_coef, cluster_count, changed, prev_model)
        else:
            assert False

    # print ('assign: ', assigns)

    return (src_model, assign_model, assigns)


if __name__ == '__main__':

    assert conf.data.startswith('data')

    series_count = conf.gridYdim * conf.gridXdim

    #1000*0.05 = 50 --> Anzahl Cluster
    cluster_count = int(series_count * conf.infFrac)
    border_distance=50

    #(np.random.uniform(low, high, size))
    #create changepoints of given amount for time points between border and amount of time steps-border
    change_pointsA = np.random.uniform(0+border_distance, conf.Nsample - border_distance, conf.Nevt).astype('int')
    change_pointsB = np.random.uniform(0+border_distance, conf.Nsample - border_distance, conf.NevtB).astype('int')

    change_points = pd.DataFrame(change_pointsA, columns=['i'])
    change_points['t'] = 'A'

    change_pointsB = pd.DataFrame(change_pointsB, columns=['i'])
    change_pointsB['t'] = 'B'

    change_points = change_points.append(change_pointsB)
    change_points.sort_values(by=['i'], inplace=True)
    change_points.reset_index(inplace=True, drop=True)

    corr_coef = []
    current_coef = INITAL_CORR_COEF

    # assign cluster distances according to input
    for i in range (0, len(change_points['t'])):
        # if it's a type A change point, then don't assign a new distance of clusters
        if change_points['t'][i] == 'A':
            corr_coef.append(current_coef)
            continue
        rand = random.choice([-1,1])
        current_coef = current_coef + rand * conf.distChange
        if current_coef > 0.7 or current_coef < 0:
            current_coef -= 2 * rand * conf.distChange
        corr_coef.append(current_coef)
    change_points['corr_coef'] = corr_coef

    if conf.infChangeFrac == -1:
        change_points['c'] = np.random.rand(len(change_points))
    else:
        change_points['c'] = np.repeat(conf.infChangeFrac, len(change_points))



    ranges = np.concatenate(([0], change_points.i))
    ranges = np.concatenate((ranges, [conf.Nsample]))
    print(ranges)

    df = pd.DataFrame()
    model = None
    assign_model = None
    assigns = None
    from utils import tqdm

    #create data according to a certain model until next change point
    for i in tqdm(range(1, len(ranges))):

        model, assign_model, assigns = generate_model(change_points.corr_coef[i-2] if i > 1 else INITAL_CORR_COEF,
                                                      ranges[i-1]-ranges[i-2], model,
                                                      assign_model,
                                                      assigns,
                                                      change_points.t[i-2] if i > 1 else '',
                                                      change_points.c[i-2] if i > 1 else 1)
        dat = generate_data(model, assigns, ranges[i] - ranges[i-1])

        df = df.append(pd.DataFrame(dat),ignore_index=True)

    start = pd.to_datetime('2000-01-01 00:00')
    df.index = pd.DatetimeIndex([start + pd.to_timedelta(5*i, unit='m') for i in df.index], name='i')

    change_points['time'] = df.index[change_points['i']]
    print(change_points)

    print('writing')
    store = pd.HDFStore(conf.wd + conf.data + '.h5')
    store['df'] = df
    store['change_points'] = change_points
    store.close()