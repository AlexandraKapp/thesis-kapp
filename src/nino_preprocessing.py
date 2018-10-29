import numpy as np
import netCDF4
import pandas as pd
import datetime
import calendar
import time

# data online source: https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.surface.html

# to convert date "hours from 1800" to unix time
# 170 years * seconds/year + leap years * seconds per day
SECS_1800_TO_1970 = (170 * 31536000) + 41 * 86400

#location of nc files
nc_dir = '../data/nino_data/'
output_file = '../data/nino.h5'

def create_dataframe_from_nc_file(year):
    """change nc file format to pd.DataFrame and preprocess data: delete grip points at pole, delete leap days
    Arguments:
        year(int): calendar year to be converted
    Return:
        pd.DataFrame: converted DataFrame
    """

    current_file = 'air.sig995.' + str(year) + '.nc'
    print(current_file)

    nc = netCDF4.Dataset(nc_dir + current_file)

    air = nc.variables['air'][:]
    time = nc.variables['time'][:]

    # delete 244 pole values
    # delete all 288 grid points from the poles, according to Wiedermann et. al
    no_poles = np.delete(air, 0, 1)
    no_poles = np.delete(no_poles, no_poles.shape[1] - 1, 1)

    # bring the matrix to the form: row=grid point; column=time stamp
    air_t = no_poles.transpose()

    flattened = air_t[0]

    for i in range(1, air_t.shape[0]):
        flattened = np.append(flattened, air_t[i], axis=0)

    # transpose to fit the input layout
    final_matrix = flattened.transpose()

    df_temp = pd.DataFrame()
    df_temp = df_temp.append(pd.DataFrame(final_matrix), ignore_index=True)

    # convert date "hours from 1800" to unix time
    new_time = (time * 3600) - SECS_1800_TO_1970

    converted_to_unix = []
    for t in new_time:
        converted_to_unix.append(datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=t))

    df_temp.index = pd.DatetimeIndex(converted_to_unix, name='i')

    if (calendar.isleap(year)):
        df_temp = df_temp.drop([datetime.datetime(year, 2, 29)])

    #df = df.append(df_temp)

    return df_temp



def anomalize_dates(df, mean_start_year, mean_end_year, anomalize_start_year, anomalize_end_year):
    """anomalize data according to ONI
    (subtracting from the time series at every grid point the long-term annual cycle
    computed over the same 30 year base periods as above that are updated every 5 years)
    Arguments:
        df (pd.DataFrame): data to be anomalized
        mean_start_year (int): start year for mean the data is anomalized with
        mean_end_year (int): end year for mean the data is anomalized with
        anomalize_start_year (int): start year of data that shall be anomalized
        anomalize_end_year (int): end year of data that shall be anomalized
    Return:
        pd.DataFrame: anomalized data for specified period
    """
    anomalized_temp = pd.DataFrame()

    # create arrays with correct days to be added for regular and leap years
    days = np.arange(np.timedelta64(0, 'D'), np.timedelta64(365, 'D'))
    leap_days = np.arange(np.timedelta64(0, 'D'), np.timedelta64(365, 'D'))

    for following_leap_day in range(59, 365):
        leap_days[following_leap_day] = leap_days[following_leap_day] + 1

    print("start of anomalizing time period: ", anomalize_start_year, "-", anomalize_end_year)

    same_days = np.ndarray(shape=(365, 0), dtype='datetime64[D]')

    # get an 2darray with all dates in one row that will create one mean
    for count in range(0, mean_end_year - mean_start_year + 1):
        current_days = days
        current_year = mean_start_year + count
        if (calendar.isleap(current_year)):
            current_days = leap_days

        same_days = np.insert(same_days, count, np.datetime64(str(current_year)) + current_days, axis=1)

    # calculate means for every grid point over mean-start to mean-end year for every single day
    means = pd.DataFrame()
    for i in range(0, 365):
        means = means.append(df.loc[same_days[i]].mean(), ignore_index=True)

    # anomalize for every grid point over anomalize-start to anomalize-end year for every single day

    for year in range(anomalize_start_year, anomalize_end_year + 1):
        anomalized_temp = anomalized_temp.append(
            df.loc[datetime.datetime(year, 1, 1):datetime.datetime(year, 12, 31)]
            .subtract(means.values))

    print('date range ', anomalize_start_year, '-', anomalize_end_year, 'anomalized over average: ', mean_start_year,
          '-', mean_end_year)
    return anomalized_temp



def create_h5file(df, file_name):
    """create h5 input file for algorithm
    Arguments:
        df(pd.DataFrame): data to be stored in h5 File
        file_name (String): Filename
    """
    store = pd.HDFStore(file_name)

    #create change points (EP El Nino years) according to Wiedermann et al.
    change_points = pd.DataFrame(columns=('i', 't', 'c', 'time'))
    change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(1958, 1, 1)), 't': 'A', 'c': 0.6,
                                          'time': datetime.datetime(1958, 1, 1)}, ignore_index=True)
    """change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(1966, 1, 1)), 't': 'A', 'c': 0.6,
                                          'time': datetime.datetime(1966, 1, 1)}, ignore_index=True)
    change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(1973, 1, 1)), 't': 'A', 'c': 0.6,
                                          'time': datetime.datetime(1973, 1, 1)}, ignore_index=True)
    change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(1977, 1, 1)), 't': 'A', 'c': 0.6,
                                          'time': datetime.datetime(1977, 1, 1)}, ignore_index=True)
    change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(1983, 1, 1)), 't': 'A', 'c': 0.6,
                                          'time': datetime.datetime(1983, 1, 1)}, ignore_index=True)
    change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(1998, 1, 1)), 't': 'A', 'c': 0.6,
                                          'time': datetime.datetime(1998, 1, 1)}, ignore_index=True)
"""
    store['df'] = df
    store['change_points'] = change_points
    store.close()

if __name__ == '__main__':
    print('reading files...')

    df = pd.DataFrame()
    for j in range(1951, 2016):
        df = df.append(create_dataframe_from_nc_file(j))

    anomalized = pd.DataFrame()

    anomalized = anomalized.append(anomalize_dates(df, 1951, 1980, 1951, 1969))
    anomalized = anomalized.append(anomalize_dates(df, 1956, 1985, 1970, 1974))
    anomalized = anomalized.append(anomalize_dates(df, 1961, 1990, 1975, 1979))
    anomalized = anomalized.append(anomalize_dates(df, 1966, 1995, 1980, 1984))
    anomalized = anomalized.append(anomalize_dates(df, 1971, 2000, 1985, 1989))
    anomalized = anomalized.append(anomalize_dates(df, 1976, 2005, 1990, 1994))
    anomalized = anomalized.append(anomalize_dates(df, 1981, 2010, 1995, 1999))
    anomalized = anomalized.append(anomalize_dates(df, 1986, 2015, 2000, 2014))

    print('Final Matrix shape: ', anomalized.shape)

    create_h5file(anomalized, output_file)
    print (output_file + ' was created.')