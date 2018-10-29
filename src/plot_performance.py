# @Patrick Siegler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

NAME = 'scale/perfBC'

infile = NAME
columns = ['run', 'Number of series', 'nsample', 'src', 'srcfrac', 'noise', 'changefrac', 'nevt', 'wnd', 'tstep', 'stat', 'time']

data = pd.read_csv('results/'+infile+'.log', sep=' ', index_col=False, names=columns)
data.drop(['run', 'srcfrac', 'changefrac', 'nevt'], axis=1, inplace=True) #TODO assumes equal / irrelevant
data = data[data['stat'].isin(['corrnorm', 'clt_clustering', 'clt_tracing', 'clt_simplify', 'clt_measure_simpl'])]
data = data[data['noise'] < 0.7]

print(data.stat.unique())
#data.stat.replace(data.stat.unique(), np.arange(len(data.stat.unique()))*10, inplace=True)


#data = data.groupby(['wnd', 'tstep', 'Number of series', 'nsample', 'stat']).mean()
#print(data)
#exit(0)


#print(data['stat'])
#plt.scatter(data['by'], data['time'], c=data['stat'], cmap=matplotlib.cm.coolwarm)

data1 = data.groupby(['stat', 'noise', 'Number of series']).mean()
stats = data1.index.levels[0].values
for s in stats:
    d = data1.ix[s].reset_index().pivot(index='Number of series', columns='noise', values='time')
    d.plot(color=['#FF0000', '#FF7000', '#FFE600', '#A1FF00', '#00FF44'], figsize=np.multiply((4,2.8),2), linewidth=2)
    #plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.98, hspace=0.4, wspace=0.4)
    #plt.figure(figsize=np.multiply((3,2),4))
    plt.savefig('thesis/img/scale/bc_'+s+'.pdf', bbox_inches="tight")

data2 = data.groupby(['noise', 'stat', 'Number of series']).mean()
data2.drop('corrnorm', level=1, inplace=True)
noises = data2.index.levels[0].values
for n in noises:
    d = data2.ix[n].reset_index().pivot(index='Number of series', columns='stat', values='time')
    for ns in d.index:
        d.ix[ns] = d.ix[ns] / d.ix[ns].sum()
    print(d.columns)
    d.columns = ['COREQ', 'Scoring', 'Simplfication', 'Tracing']
    ax = d.ix[:,['Scoring', 'Simplfication', 'Tracing', 'COREQ']].plot.bar(stacked=True, ylim=(0,1), legend='reverse', figsize=np.multiply((4,2.5),2))
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=0)
    plt.savefig('thesis/img/scale/bc_fractions_{:02.0f}.pdf'.format(n*10), bbox_inches='tight')

#plt.show()