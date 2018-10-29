import pandas as pd
import numpy as np
import os
import csv
from scipy import stats

import conf

def write_scores(path, columns, values, eps, th, corrCoef, filename, noise, clusterAlg, wnd, tstep):
    """write scores to csv file
    """
    with open(path, 'a+', newline='\n') as csv_file:
        writer = csv.writer(csv_file)
        if os.stat(path).st_size == 0:
            header = ['Filename', 'Noise', 'Cluster Alg','Eps', 'Threshold', 'type B CorrCoef', 'Window', 'time step']
            header.extend(columns)
            writer.writerow(header)
        print_line = pd.Series([filename, noise, clusterAlg, eps, th, corrCoef, wnd, tstep]).append(values)
        writer.writerow(print_line)


def compute_CoD(df, columns, th):
    """compute coefficient of determination based on model
    Arguments:
        df(DataFrame): each sorces values for every time step
        columns(list): all column(respective score) names
        th(float): threshold
    Return:
        pd.Series: coefficient of determination for each score
    """

    model = df[np.abs(df - np.mean(df)) > th * np.std(df)]
    model.fillna(0, inplace=True)
    df_c = df.copy()
    df_c.fillna(0, inplace=True)

    r_values = []
    for col in columns:
        if col not in df_c.columns:
            r_values.append(None)
            continue
        slope, intercept, r_value, p_value, std_err = stats.linregress(model[col], df_c[col])
        r_values.append(np.square(r_value))
    return pd.Series(r_values)

def compute_recPrecF1_allTh(df, columns, change_points, filename, noise, eps, corrCoef, clusterAlg, wnd, tstep):
    """calcualate the precision, recall and f1 score for all thresholds from 0.5 through 3
    """
    for th in [0.5, 1, 1.5, 2, 2.5, 3]:
        logA = pd.DataFrame(index=['precision', 'recall', 'f1'])
        logB = pd.DataFrame(index=['precision', 'recall', 'f1'])
        logALL = pd.DataFrame(index=['precision', 'recall', 'f1'])

        for col in columns:

            if col not in df.columns:
                logA[col] = (None, None, None)
                logB[col] = (None, None, None)
                continue

            hitsALL = 0
            # get all positions where the distance to the mean is more than th * standard deviation
            points = np.where(np.abs(df[col] - np.mean(df[col])) > th * np.std(df[col]))[0]

            points *= tstep
            hitsA = 0
            index_change_pointsA = change_points[change_points.t == 'A'].iloc[:, 0]

            # make sure, one hit is only counted for one change point and not two, otherwise precision can be > 1
            cp_not_in_same_window_as_prev = np.diff(index_change_pointsA) > wnd
            index_change_pointsA = np.array(index_change_pointsA)[np.append(True, [cp_not_in_same_window_as_prev])]

            for cp in index_change_pointsA:
                if np.any(np.logical_and(0 < cp - points, cp - points < wnd * 2)):
                    hitsA += 1
                    hitsALL += 1

            hitsB = 0
            index_change_pointsB = change_points[change_points.t == 'B'].iloc[:, 0]
            for cp in index_change_pointsB:
                if np.any(np.logical_and(0 < cp - points, cp - points < wnd * 2)):
                    hitsB += 1
                    hitsALL += 1

            logA[col] = get_rec_prec_f1(hitsA, len(points), len(index_change_pointsA))
            logB[col] = get_rec_prec_f1(hitsB, len(points), len(index_change_pointsB))
            logALL[col] = get_rec_prec_f1(hitsALL, len(points), len(change_points.iloc[:, 0]))

        write_scores(conf.evaluationDir + '\\precA.csv', columns, logA.loc['precision'], eps, th, corrCoef, filename,
                     noise, clusterAlg, wnd, tstep)
        write_scores(conf.evaluationDir + '\\recA.csv', columns, logA.loc['recall'], eps, th, corrCoef, filename,
                     noise, clusterAlg, wnd, tstep)
        write_scores(conf.evaluationDir + '\\f1A.csv', columns, logA.loc['f1'], eps, th, corrCoef, filename, noise,
                     clusterAlg, wnd, tstep)
        write_scores(conf.evaluationDir + '\\precB.csv', columns, logB.loc['precision'], eps, th, corrCoef, filename,
                     noise, clusterAlg, wnd, tstep)
        write_scores(conf.evaluationDir + '\\recB.csv', columns, logB.loc['recall'], eps, th, corrCoef, filename,
                     noise, clusterAlg, wnd, tstep)
        write_scores(conf.evaluationDir + '\\f1B.csv', columns, logB.loc['f1'], eps, th, corrCoef, filename, noise,
                     clusterAlg, wnd, tstep)
        write_scores(conf.evaluationDir + '\\precALL.csv', columns, logALL.loc['precision'], eps, th, corrCoef,
                     filename, noise, clusterAlg, wnd, tstep)
        write_scores(conf.evaluationDir + '\\recALL.csv', columns, logALL.loc['recall'], eps, th, corrCoef, filename,
                     noise, clusterAlg, wnd, tstep)
        write_scores(conf.evaluationDir + '\\f1ALL.csv', columns, logALL.loc['f1'], eps, th, corrCoef, filename,
                     noise, clusterAlg, wnd, tstep)


def get_rec_prec_f1(hits, detections_count, cp_count):
    """ compute recall, precision and f1 score
    Arguments:
        hits(int): amount of correctly detected change events
        detections: amount of detected changes
        cp_count: amount of actual change points
    """
    if detections_count > 0:
        prec = hits / detections_count
    else:
        prec = 0
    if cp_count > 0:
        rec = hits/ cp_count

        if (prec + rec) > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0
    else:
        rec = None
        f1 = None
    return prec, rec, f1

def compute_CoD_allTh(df, columns, filename, noise, eps, corrCoef, clusterAlg, wnd, tstep):
    """calcualate the coefficient of determination for all thresholds from 0.5 through 3
    """
    for th in (0.5, 1, 1.5, 2, 2.5, 3):
        r_values = compute_CoD(df, columns, th)
        write_scores(conf.evaluationDir + '\\coef_of_determination.csv',
                     columns, r_values, eps, th, corrCoef, filename, noise, clusterAlg, wnd, tstep)