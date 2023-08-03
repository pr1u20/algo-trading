# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:23:56 2023

@author: Pedro
"""

from alpaca.data.live import StockDataStream, CryptoDataStream
import config
from alpaca.trading.client import TradingClient

import nest_asyncio
nest_asyncio.apply()

def market_open():

    client = TradingClient(config.API_KEY, config.SECRET_KEY)
    
    clock = client.get_clock()
    
    if clock.is_open:
        print('The market is {}'.format('open.'))
              
    else: 
        print('The market is {}'.format('closed.'))
        time_for_open = (clock.next_open - clock.timestamp)
        secs = time_for_open.total_seconds()
        hours = int(secs / 3600)
        minutes = int(secs / 60) % 60
        print(f'Opens in {hours} h {minutes} m ({time_for_open})')
    
    return clock

def real_time_stock(symbol = "TSLA"):
    
    wss_client = StockDataStream(config.API_KEY, config.SECRET_KEY)
    
    # async handler
    async def bars_data_handler(data):
        # real-time data will be displayed here
        # as it arrives
        print(data)
        print("===")
    
    wss_client.subscribe_bars(bars_data_handler, symbol)
    
    wss_client.run()
    
def real_time_crypto(symbol = "BTC/USD"):
    
    wss_client = CryptoDataStream(config.API_KEY, config.SECRET_KEY)
    
    # async handler
    async def bars_data_handler(data):
        # real-time data will be displayed here
        # as it arrives
        print(data)
        print("===")
    
    wss_client.subscribe_bars(bars_data_handler, symbol)
    
    wss_client.run()
    
    

if __name__ == "__main__":
    
    clock = market_open()

    if clock.is_open:
        
        real_time_stock()
    
    real_time_crypto()