import numpy as np
import pandas as pd
import datetime

"""Stock data is preprocessed by droping all stocks with missing values, 
using the log growth rate and change points for financial crises are created.
Everything is stored in a h5 file, usable for clt.py
"""

FILENAME_INPUT = '../data/stocks/SP500.csv'
FILENAME_OUTPUT = '../data/stocks/SP_withoutNull_LogGrowth.h5'


def store_h5(file, df, change_points):
    """store dataframes to h5 file
    Arguments:
        file (String): filename where file shall be stored
        df (pd.DataFrame): data to be stored
        change_points(pd.DataFrame): change points to be stored
    """

    store = pd.HDFStore(file)
    store['df'] = df
    store['change_points'] = change_points
    store.close()


def get_change_points(df):
    """create all financial crises as change points
    Arguments:
        df(pd.DataFrame): data the change points are created for. needed to get the right index for the change points
    Return:
        pd.DataFrame: change points
    """
    change_points = pd.DataFrame(columns=('i', 't', 'c', 'time'))

    # https://en.wikipedia.org/wiki/List_of_stock_market_crashes_and_bear_markets
    # financial crises according to https://en.wikipedia.org/wiki/Financial_crisis
    # Dot-com bubble Collapse of a technology bubble.
    change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(2000, 3, 10)), 't': 'A', 'c': 0.6,
                                          'time': datetime.datetime(2000, 3, 10)}, ignore_index=True)

    # Economic effects arising from the September 11 attacks
    # no prices for 11 through 16 sept existing
    change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(2001, 9, 17)), 't': 'A', 'c': 0.6,
                                          'time': datetime.datetime(2001, 9, 17)}, ignore_index=True)

    # Stock market downturn of 2002 (United States, Canada, Asia, and Europe)
    # After recovering from lows reached following the September 11 attacks, indices slid steadily starting in March 2002, with dramatic declines in July and September
    change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(2002, 10, 9)), 't': 'A', 'c': 0.6,
                                          'time': datetime.datetime(2002, 10, 9)}, ignore_index=True)

    # Chinese stock bubble of 2007
    # The SSE Composite Index of the Shanghai Stock Exchange tumbles 9% from unexpected selloffs, the
    # largest drop in 10 years, triggering major drops in worldwide stock markets.
    # change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(2007, 2, 27)), 't': 'A', 'c': 0.6,
    #                                     'time': datetime.datetime(2007, 2, 27)}, ignore_index=True)

    # United States bear market of 2007–09
    # Till June 2009, the Dow Jones Industrial Average, Nasdaq Composite and S&P 500
    # all experienced declines of greater than 20% from their peaks in late 2007.
    change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(2007, 10, 11)), 't': 'A', 'c': 0.6,
                                          'time': datetime.datetime(2007, 10, 11)}, ignore_index=True)

    # Financial crisis of 2007–08
    # On September 16, 2008, failures of large financial institutions in the United States,
    # due primarily to exposure of securities of packaged subprime loans and credit default swaps
    # issued to insure these loans and their issuers, rapidly devolved into a
    # global crisis resulting in a number of bank failures in Europe and sharp
    # reductions in the value of equities (stock) and commodities worldwide.
    change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(2008, 9, 16)), 't': 'A', 'c': 0.6,
                                          'time': datetime.datetime(2008, 9, 16)}, ignore_index=True)

    # 2009 Dubai debt standstill
    # Dubai requests a debt deferment following its massive renovation and development projects,
    # as well as the Great Recession. The announcement causes global stock markets to drop.
    # change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(2009, 11, 27)), 't': 'A', 'c': 0.6,
    #                                      'time': datetime.datetime(2009, 11, 27)}, ignore_index=True)

    # European sovereign debt crisis
    # Standard & Poor's downgrades Greece's sovereign credit rating to junk four days after
    # the activation of a €45-billion EU–IMF bailout,
    # triggering the decline of stock markets worldwide and of the Euro's value,
    # and furthering a European sovereign debt crisis.
    change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(2010, 4, 27)), 't': 'A', 'c': 0.6,
                                          'time': datetime.datetime(2010, 4, 27)}, ignore_index=True)

    # 2010 Flash Crash
    # The Dow Jones Industrial Average suffers its worst intra-day point loss,
    # dropping nearly 1,000 points before partially recovering.
    change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(2010, 5, 6)), 't': 'A', 'c': 0.6,
                                          'time': datetime.datetime(2010, 5, 6)}, ignore_index=True)

    # August 2011 stock markets fall
    # S&P 500 entered a short-lived bear market between 02nd May 2011 (intraday high: 1,370.58)
    # and 04 October 2011 (intraday low: 1,074.77),
    # a decline of 21.58%. The stock market rebounded thereafter and ended the year flat.
    change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(2011, 8, 1)), 't': 'A', 'c': 0.6,
                                          'time': datetime.datetime(2011, 8, 1)}, ignore_index=True)

    # 2015–16 Chinese stock market crash
    # China stock market crash starts in June and continues into July and August. In January 2016,
    # Chinese stock market experiences a steep sell-off which sets off a global rout.
    # change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(2015, 6, 12)), 't': 'A', 'c': 0.6,
    #                                      'time': datetime.datetime(2015, 6, 12)}, ignore_index=True)

    # 2015–16 stock market selloff
    # The Dow Jones fell 588 points during a two-day period, 1,300 points from August 18–21. On Monday, August 24,
    # world stock markets were down substantially, wiping out all gains made in 2015
    change_points = change_points.append({'i': df.index.get_loc(datetime.datetime(2015, 8, 18)), 't': 'A', 'c': 0.6,
                                          'time': datetime.datetime(2015, 8, 18)}, ignore_index=True)

    return change_points


if __name__ == '__main__':
    df = pd.read_csv(FILENAME_INPUT)
    df.set_index('date', inplace=True)
    df.index = pd.DatetimeIndex(df.index)

    # drop rows with all NaN values
    df.dropna(how='all', inplace=True)

    # preprocessing of stock data: log growth rate
    df_GrowthLog = (np.log(df) - np.log(df).shift(1))
    df_GrowthLog.dropna(how='all', inplace=True)

    without_nullLog = df_GrowthLog.dropna(axis=1)

    store_h5(FILENAME_OUTPUT, without_nullLog, get_change_points(without_nullLog))