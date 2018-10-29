import argparse
import sys


class ConfArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        if len(arg_line) == 0 or arg_line[0] == '#':
            return []
        elif arg_line.startswith('--'):
            return [arg_line]
        else:
            line = arg_line.partition('=')
            return ['--' + line[0], line[2]]


parser = ConfArgumentParser(fromfile_prefix_chars='@')

dataset = parser.add_argument_group(title='dataset')
dataset.add_argument('--wd', required=True, help="Location of data files")
dataset.add_argument('--data', required=True, help='Column to be evaluated')
dataset.add_argument('--gridXdim', type=int, required=True, help='Size of dimension X of grid in dataset, from preprocessing')
dataset.add_argument('--gridYdim', type=int, required=True, help='Size of dimension Y of grid in dataset, from preprocessing')
dataset.add_argument('--diff', action='store_true', help="Difference the data while loading (it is autocorrelated)")


parser.add_argument('--corr', action='store_true', help='Use correlation as distance measure rather than euclidean')

scope = parser.add_argument_group(title='scope', description='if t and w are supplied that specific time will be considered, otherwise the time span tstart-tend is considered (in a sliding window w, if supplied)')
scope.add_argument('--gridXmin', type=int, help='Subgrid to act on, all data if not provided')
scope.add_argument('--gridXmax', type=int, help='Subgrid to act on, all data if not provided')
scope.add_argument('--gridYmin', type=int, help='Subgrid to act on, all data if not provided')
scope.add_argument('--gridYmax', type=int, help='Subgrid to act on, all data if not provided')
scope.add_argument('--t', help='Point in time to act on')
scope.add_argument('--wnd', type=int, help='Time window to consider')
scope.add_argument('--tstart', help='Start point to act on')
scope.add_argument('--tend', help='End point to act on')
scope.add_argument('--tstep', type=int, default=1, help='Sliding window step size')

clustering = parser.add_argument_group(title='clustering')
clustering.add_argument('--clustering', choices=['DBSCAN', 'FastDBSCAN', 'COREQ'], default='DBSCAN', help='Clustering algorithm to use')
clustering.add_argument('--eps', type=float, required=True, help='eps for DBSCAN (upper limit for MultiDBSCAN), alpha for COREQ (default 0.71)')
clustering.add_argument('--epsDelta', type=float, help='MultiDBSCAN eps step size')
clustering.add_argument('--epsLower', type=float, help='MultiDBSCAN eps lower limit')
clustering.add_argument('--minPts', type=int, default=1, help='minPts for DBSCAN, kappa for COREQ')

chen = parser.add_argument_group(title='chen')
chen.add_argument('--chenTh', type=float, help='threshold for Chen14 event detection')
chen.add_argument('--chenNoHits', action='store_true', help='Disable plotting the detected events')
chen.add_argument('--chenPlotClusterOnly', action='store_true', help='for each detected event, plot only the forming or disbanding cluster, rather than all streams')

clt = parser.add_argument_group(title='tracing')
clt.add_argument('--cltNoRender', action='store_true', help='Disable rendering the graph to dot')
clt.add_argument('--cltNoStats', action='store_true', help='Disable plotting the metrics')
clt.add_argument('--cltPlotFullStats', action='store_true', help='Plot each statistic in a separate file (if not --cltNoRender)')
clt.add_argument('--cltIncludeNorm', action='store_true', help='Include matrix difference norm in scoring')
clt.add_argument('--cltIncludeChen', action='store_true', help='Also run Chen and include it in scoring')
clt.add_argument('--cltChenEps', type=float, help='upper limit for MultiDBSCAN of eps for Chen in clt')
clt.add_argument('--cltChenEpsDelta', type=float, help='MultiDBSCAN eps step size for Chen in clt')
clt.add_argument('--cltChenEpsLower', type=float, help='MultiDBSCAN eps lower limit for Chen in clt')

generate = parser.add_argument_group(title='generation')
generate.add_argument('--Nsample', type=int, default=10000, help='number of time steps/samples to be generated')
generate.add_argument('--Nevt', type=int, default=3, help='number of global change events (type A, cluster switch) to be generated')
generate.add_argument('--infMethod', choices=['uniform', 'dirichlet'], default='dirichlet', help='Inflate: How series are assigned to an identity')
generate.add_argument('--infFrac', type=float, default=0.1, help='Inflate: Number of cluster identities as fraction of number of series (0, 1)')
generate.add_argument('--infNoise', type=float, default=0.1, help='Inflate: Amount of noise of inflated series')
generate.add_argument('--infChangeFrac', type=float, default=1, help='Inflate: Amount of change (0, 1) or -1 for uniform random')
generate.add_argument('--distChange', type=float, default=0.5,  help='Correlation coefficient change between clusters at an inter-cluster event')
generate.add_argument('--NevtB', type=int, default=0, help='Inflate: number of global change events (type B, cluster drift) to be generated')

parser.add_argument('--plotFile', action='store_true', help='plot to file instead of showing windows')
parser.add_argument('--plotDir', default="plots", help="Location of rendered plots (--plotFile")
parser.add_argument('--evaluationDir', default="evaluation", help="Location of evaluation files")
parser.add_argument('--nJobs', type=int, default=-1, help='number of jobs for parallel operations')
parser.add_argument('--progress', action='store_true', help='print progress meters')
parser.add_argument('--verbose', action='store_true', help='print details what is being done')
parser.add_argument('--storeClusterAssign', action='store_true', help='store the cluster assignment of time series to csv file')
parser.add_argument('--storeScores', action='store_true', help='store score values to csv file')

args = parser.parse_args(sys.argv[1:])

if args.gridXmin is not None or args.gridXmax is not None or args.gridYmin is not None or args.gridYmax is not None:
    if args.gridXmin is None or args.gridXmax is None or args.gridYmin is None or args.gridYmax is None:
        parser.error("Either supply all or none of grid(X/Y)(max/min)")
else:
    assert args.data != "call" and args.data != "sms" #those are 1-indexed
    args.gridXmin = 0
    args.gridXmax = args.gridXdim - 1
    args.gridYmin = 0
    args.gridYmax = args.gridYdim - 1

if args.t is None or args.wnd is None:
    if args.tstart is None or args.tend is None:
        parser.error("Either t and w or tstart and tend are required")

if (args.epsDelta and args.epsLower is None) or (args.epsLower and args.epsDelta is None):
    parser.error("Supply either both epsDelta and epsLower, or neither")

if args.clustering == 'COREQ':
    if args.eps >= 1: #or args.epsDelta is not None or args.epsLower is not None:
        parser.error("For COREQ, eps must be < 1")
    if args.corr != True:
        parser.error("COREQ requires corr")
else:
    if args.eps < 1 or args.epsLower is not None and args.epsLower < 1:
        pass
        #parser.error("Eps < 1 is reserved for COREQ (for file name discrimination only, feel free to comment this)")

if args.cltIncludeChen:
    if args.cltChenEps is None or args.cltChenEpsDelta is None or args.cltChenEpsLower is None:
        parser.error('If --cltIncludeChen, then cltChenEps, -EpsDelta and -EpsLower must be supplied')

# hacky: Write arguments to global scope, so that other modules can use conf.arg
for x in args.__dict__:
    sys.modules[__name__].__dict__[x] = args.__dict__[x]