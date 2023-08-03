# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 11:39:54 2023

@author: Pedro
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import pandas_ta as ta
import numpy as np

from market_data import get_data_binance, load_data
from training_data import create_indicators
from lightweight_charts import Chart
from time import sleep

class Graph():
    
    """
    graph = Graph(df, random_list)
    
    graph.buy(np.array(df['date'][20:20+1]), np.array(df['open'][20:20+1]))
    
    graph.sell(np.array(df['date'][40:40+1]), np.array(df['open'][40:40+1]))
    
    graph.show()
    """
    
    def __init__(self, df, indicator):
        
        self.fig = make_subplots(rows=4, cols=1, row_width=[0.2, 0.1, 0.1, 0.6])
        
        self.fig.add_trace(go.Candlestick(x=df['time'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close']), row = 1, col = 1)
        
        #self.fig.update_xaxes(rangeslider_thickness = 0.1)
        #self.fig.update_xaxes(rangeslider_visible=False)
        
        self.fig.add_trace(go.Bar(x=df['time'], 
                                      y=df['volume'],
                                      name='volume'), 
                           row = 3, 
                           col = 1)
        
        self.fig.add_trace(go.Scatter(x=df['time'], 
                                      y=indicator,
                                      mode='lines',
                                      name='indicator'), 
                           row = 4, 
                           col = 1)
        
        self.fig.update_xaxes(matches='x')
        
    def buy(self, x, y):
        
        color = "#00FF00"
            
        self.fig.add_trace(go.Scatter(x=x, 
                                      y=y- 0.05*y,
                                      mode='markers',
                                      name='buy orders',
                                      marker_symbol = "triangle-down",
                                      marker_color = color,
                                      marker_size = 10), 
                           row = 1, 
                           col = 1)
        
    def sell(self, x, y):
        
        color = "#FF0000"
            
        self.fig.add_trace(go.Scatter(x=x, 
                                      y=y- 0.05*y,
                                      mode='markers',
                                      name='buy orders',
                                      marker_symbol = "triangle-down",
                                      marker_color = color,
                                      marker_size = 10), 
                           row = 1, 
                           col = 1)
        
        
    def show(self):
        self.fig.show()


class GraphApp():
    def __init__(self, data):
        self.data = data
        self.chart = Chart(width = 1920, height = 1080, inner_width=1, inner_height=0.7)
        self.chart.set(self.data[['open', 'high', 'low', 'close', 'time', 'volume']])
        self.chart.legend(visible=True, font_size=200)
        self.used_indicators = []
        self.first_subchart_created = False
        self.second_subchart_created = False
        self.line_object_dictionary = {}
        
    def buy(self, time, price):
        
        self.chart.marker(time, text = f"Buy @ {round(price)}", position = 'below', shape = "arrow_up", color = "#00FF00")
    
    def sell(self, time, price):
        
        self.chart.marker(time, text = f"Sell @ {round(price)}", position = 'above', shape = "arrow_down", color = "#FF0000")
        
    def indicator(self, type_str, color):
        
        self.data[type_str].fillna(self.data[type_str].mean(), inplace= True)
        
        line = self.chart.create_line(color = color)
        
        self.data.rename(columns = {type_str:'value'}, inplace = True)
        line.set(self.data[['time', 'value']])
        self.data.rename(columns = {'value':type_str}, inplace = True)
        
        self.line_object_dictionary[type_str] = line
        
    
    def indicator_subchart(self, type_str, color = "blue", subchart_number = 1):
        if subchart_number == 1:
            if self.first_subchart_created == False:
                self.indicator_chart_1 = self.chart.create_subchart(width=1, height=0.15, sync=True, volume_enabled=False)
                self.first_subchart_created = True
                
            line = self.indicator_chart_1.create_line(color = color)
            
        elif subchart_number == 2:
            if self.second_subchart_created == False:
                self.indicator_chart_2 = self.chart.create_subchart(width=1, height=0.15, sync=True, volume_enabled=False)
                self.second_subchart_created = True
                
            line = self.indicator_chart_2.create_line(color = color)
        
        self.data[type_str].fillna(self.data[type_str].mean(), inplace= True)
        
        self.data.rename(columns = {type_str:'value'}, inplace = True)
        line.set(self.data[['time', 'value']])
        self.data.rename(columns = {'value':type_str}, inplace = True)
        
        self.line_object_dictionary[type_str] = line
    
    def rsi(self):
        
        period = 14
        self.data['rsi'] = self.data.ta.rsi()
        self.data['rsi'].fillna(self.data['rsi'].iloc[period], inplace= True)
    
        indicator_chart = self.chart.create_subchart(width=1, height=0.15, sync=True, volume_enabled=False)
        line = indicator_chart.create_line()
        
        self.data.rename(columns = {'rsi':'value'}, inplace = True)
        line.set(self.data[['time', 'value']])
        self.data.rename(columns = {'value':'rsi'}, inplace = True)
    
        
    
    def indicators_MA(self, type_str = 'MA20', color = "blue"):
        
        period = int(type_str[2:])
        
        self.data[type_str] = self.data['close'].rolling(window=period).mean()
        self.data[type_str].fillna(self.data[type_str].iloc[period], inplace= True)
        
        line = self.chart.create_line(color = color)
        
        self.data.rename(columns = {type_str:'value'}, inplace = True)
        line.set(self.data[['time', 'value']])
        self.data.rename(columns = {'value':type_str}, inplace = True)
        
        self.line_object_dictionary[type_str] = line
        
    def update_all(self, series):
        for type_str in self.line_object_dictionary:
            line = self.line_object_dictionary[type_str]
            
            series.rename(columns = {type_str:'value'}, inplace = True)
            line.update(series[['time', 'value']].squeeze())
            series.rename(columns = {'value':type_str}, inplace = True)
            
            self.line_object_dictionary[type_str] = line
        
    def set_visible_range(self, start_timestamp, end_timestamp):
        
        self.chart.run_script(f"{self.chart.id}.chart.setVisibleRange({{ from: {start_timestamp}, to: {end_timestamp} }})")
        
        self.chart.run_script(f'''
          {self.chart.id}.chart.applyOptions({{
              timeScale: {{
                  visibleRange: {
                      "from": start_timestamp,
                      "to": end_timestamp
                  }
              }}
          }})''')
         
        
    def show(self):
        
        self.chart.show()
        
    def live(self, symbol):
        
        while not self.chart._exit.is_set():
            # Load and save data from last 2 min
            success = get_data_binance(symbol = symbol, start = "5m ago", save = 'live')
            # Load data from file, to have same format as self.data
            if not isinstance(success, type(None)):
                df = load_data('live')
                for i in range(len(df)):
                    
                    new_data = df.iloc[i]
                    # Add the new data to the self.data dataframe
                    self.data.loc[df.iloc[i]['time']] = new_data
                    self.chart.update(new_data[['open', 'high', 'low', 'close', 'time', 'volume']])
                print(df)
                self.data.iloc[-101:] = create_indicators(self.data.iloc[-101:], get_labels = False)
                self.update_all(self.data.iloc[-1:])
                
            sleep(2)
        


        
        
if __name__ == "__main__":
    
    symbol = "BTCUSDT"
    #df = load_data(symbol).iloc[:1000]
    df = get_data_binance(symbol = symbol, interval='5m', start = "1d ago", end = None, save = None)
    df = create_indicators(df, get_labels = False)
    
    df['labels'] = [random.random() for i in range(len(df['time']))]
    
    graph = GraphApp(df)
    graph.indicator('MA30', color = "#add8e6")
    graph.indicator('MA100', color = "#ffffed")
    graph.buy(df['time'].iloc[-100], 40.3)
    graph.sell(df['time'].iloc[-40], 54.4)
    graph.indicator_subchart('macd', subchart_number = 2)
    graph.indicator_subchart('rsi', subchart_number = 1)
    #graph.indicator_subchart('macd')
    #graph.set_visible_range(1625508000000, 1625508200000)
    graph.show()
    from time import sleep
    sleep(2)
    graph.chart._q.put('exit')
    #graph.chart._exit.wait()
    graph.chart._process.terminate()
    #graph.live(symbol)