# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:13:03 2023

@author: Pedro
"""

import config

from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import os

import pandas as pd
from binance.client import Client
import datetime as dt
# client configuration

pd.set_option('mode.chained_assignment', None)


def get_data_alpaca(symbols = ["TSLA"], timeframe = TimeFrame.Minute, start_str = "2023-06-01", data_type = "stocks"):
    
    #bars = get_data_alpaca(symbols = ["BTC/USD"], timeframe = TimeFrame.Minute, start_str = "2023-06-19", data_type = "crypto")
    
    if data_type == "stocks":
        client = StockHistoricalDataClient(api_key = config.API_KEY, secret_key = config.SECRET_KEY)
    
        request_params = StockBarsRequest(
                            symbol_or_symbols=symbols,
                            timeframe=timeframe,
                            start=datetime.strptime(start_str, '%Y-%m-%d')
                            )
    
        bars = client.get_stock_bars(request_params)
    
        # convert to dataframe
        bars = bars.df
    
        bars['Date'] = bars.index.get_level_values(1)
        
    elif data_type == "crypto":
        
        client = CryptoHistoricalDataClient(api_key = config.API_KEY, secret_key = config.SECRET_KEY)
    
        request_params = CryptoBarsRequest(
                            symbol_or_symbols=symbols,
                            timeframe=timeframe,
                            start=datetime.strptime(start_str, '%Y-%m-%d')
                            )
    
        bars = client.get_crypto_bars(request_params, feed = "us")
    
        # convert to dataframe
        bars = bars.df
    
        bars['time'] = bars.index.get_level_values(1)
    
    return bars

def get_data_binance(symbol = "DOTUSDT", interval='5m', start = "22 June,2023", end = None, save = None):
    
    try:
        client = Client(config.API_KEY_BINANCE, config.SECRET_KEY_BINANCE)
    
        klines = client.get_historical_klines(symbol, interval, start, end)
        data = pd.DataFrame(klines)
         # create colums name
        data.columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
                    
        # change the timestamp
        data.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data.close_time]
        
        #convert data to float and plot
        data=data.astype(float)
        data['time'] = data.index
        
        if not save == None:
            file = os.path.join("data", save +'.csv')
            data.to_csv(file, index = False, header=True)
            
    except Exception as e:
        data = None
        print("[Error] Data not retrieved.")
        print(e)
    
    return data

def load_data(symbol = "DOTUSDT"):
    
    file = os.path.join("data", symbol + '.csv')
    
    data = pd.read_csv(file)
    
    data.index = data['time']
    data = process_repeated_timestamps(data)
    
    return data

def process_repeated_timestamps(df):
    
    # Find duplicates in the index and add one millisecond to each duplicate occurrence
    # Step 2: Identify duplicated indexes
    duplicates_mask = df.index.duplicated()

    # Step 3: Modify the first duplicate index by truncating the last character
    df.loc[duplicates_mask, df.index.name] = df.loc[duplicates_mask].index + '9'
        
    return df
    
    
    

if __name__ == "__main__":

    df = get_data_binance(save = "DOTUSDT")
    #data = load_data()
    
    #bars = get_data(symbols = ["BTC/USD"], timeframe = TimeFrame.Minute, start_str = "2023-06-19", data_type = "crypto")