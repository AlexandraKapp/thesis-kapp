#Monitoring Cluster Dynamics for Change Detection in Time Series Collections

Dependencies: Pandas, NumPy, scipy, scikit-learn, matplotlib, tqdm
(for querying SP500 stock data, also: alpha_vantage)

Quick Setup:
* Open the directory with PyCharm
* Run `generate_inflate.py` with argument `@conf_data` in working directory root (defaults to src/)
* Run `clt.py @conf_inflate` (again, adjust working directory). When using COREQ, you first need to run `setup.py build` in the BlockCorr folder and copy the library to the src folder

Run any script with `-h` to view the parameters you can define in the config file.

Major scripts are:
* `generate_inflate.py` generates the data according to the config
* `clt.py` runs cluster score computation, optionally including corrNorm and chen. 
    Outputs are  evaluation measures (coefficient of determination, precision, recall, f1 values) stored in csv files in the defined `evaluationDir`,  score values are logged and stored at `evaluationDir` (if `--storeScores`), the plotted scores stored in `plotDir` (if `--plotFile`) and the cluster assignment of each time series for each time step is stored as csv file in `evaluationDir` (if `--storeClusterAssign`)
* `chen.py` runs chen in the original way, optionally plotting each hit