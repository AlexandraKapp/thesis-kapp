import datetime
import pandas as pd
import time
from alpha_vantage.timeseries import TimeSeries

API_KEY = 'XXX' #insert key

#https://www.alphavantage.co/

SYMBOLS_FILE = "sp_symbols.txt"
CSV_FILE = '../data/stocks/SP500.csv'


def alphaV_stocks(symbol):
    """query stock from quandl API
        Arguments:
            symbol(String): a stock symbol to be queried, e.g. 'AAPL'
    """

    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, meta_data = ts.get_daily_adjusted(symbol=symbol,outputsize='full')
    data.index = pd.DatetimeIndex(data.index)
    return data

if __name__ == '__main__':
    stocks = open(SYMBOLS_FILE).read().splitlines()

    df = pd.DataFrame(index=pd.date_range(datetime.datetime(1995, 1, 1), datetime.datetime(2018, 1, 7), freq='B'))
    df.index.name = 'date'

    counter = -1
    start_time = time.time()
    for stock in stocks:
        counter += 1

        #API has limit of requests per minute, therefore time outs inbetween
        if counter % 3 == 0 and counter != 0:
            t = time.time() - start_time
            print(str(t))
            time.sleep(60 - t)
            start_time = time.time()

        print("current stock: ", stock, " counter: ", counter)
        try:
            df[stock] = alphaV_stocks(stock)['5. adjusted close']
        except Exception as e:
            print(e)

    df.to_csv(CSV_FILE)
