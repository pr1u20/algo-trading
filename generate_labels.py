# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 17:33:42 2023

@author: Pedro
"""

from market_data import get_data_binance, load_data

import pandas as pd
import numpy as np

labelling = [0, 1, 2, 3, 4]
percentages = np.array([- 0.5, -0.1, 0.3, 0.7])


def create_labels(df, future_prediction_range = 20):
    
    percentages = np.array([- 0.5, -0.1, 0.3, 0.7])
    
    ML_values = []
    
    for i in range(len(df) - future_prediction_range):
        
        current_price = df.iloc[i]['close']
        mean_price = df.iloc[i:i + 4]['close'].mean()
        mean_price_diff = mean_price - current_price 
        
        if mean_price_diff >= 0:
            
            max_price = df.iloc[i:i + future_prediction_range]['close'].max()
            arg_max_price = df.iloc[i:i + future_prediction_range]['close'].argmax()
            min_price = df.iloc[i:i+arg_max_price+1]['close'].min()
            mean_price = df.iloc[i:i+arg_max_price+1]['close'].mean()
            mean_price_diff = mean_price - current_price 
            
        elif mean_price_diff < 0:
            min_price = df.iloc[i:i + 5]['close'].min()
            arg_min_price = df.iloc[i:i + 5]['close'].argmin()
            max_price = df.iloc[i:i+arg_min_price+1]['close'].max()
            mean_price = df.iloc[i:i+arg_min_price+1]['close'].mean()
            mean_price_diff = mean_price - current_price 
        
        max_drawdown = 100 * (min_price - current_price + current_price*1e-7) / current_price
        max_gain = 100 * (max_price - current_price + current_price*1e-7) / current_price
        
        
        std_range = 100
        if  10 < i < 100:
            std_range = i
            
        std = (df.iloc[i - std_range:i]['close'].std() / df.iloc[i]['close']) * 100
        
        
        if std >= 1:
            percentages *= std**(1.5)
             
        if max_drawdown <= percentages[0]:
            
            value = labelling[0]
            
        elif percentages[0] <= max_drawdown <= percentages[1]: 
            value = labelling[1]
            
        elif max_drawdown >= percentages[1] and max_gain <= percentages[2]:
            value = labelling[2]
        
        elif percentages[2] <= max_gain <= percentages[3]:
            value = labelling[3]
            
        elif max_gain > percentages[3]:
            value = labelling[4]
            
        else:
            value = labelling[2]
        

        ML_values.append(value)
        
    for _ in range(future_prediction_range):
        ML_values.append(np.nan)
        
    df['labels'] = ML_values
        
    return df

def create_labels_slow(df, future_prediction_range = 20):
    
    percentages = [-0.5, 0.7]
    
    labelling = [-1, 0, 1]
    
    ML_values = []
    std_values = []
    
    for i in range(len(df) - future_prediction_range):
        
        current_price = df.iloc[i]['close']
        stop_loss  = current_price * (100 - 0.3) / 100
        previous_price = current_price
        max_price = current_price
        max_gain = 0
        max_drawdown = 0
        
        std_range = 100
        if  10 < i < 100:
            std_range = i
            
        std = (df.iloc[i - std_range:i]['close'].std() / df.iloc[i]['close']) * 100
        
        sell_limit = percentages[-1]
        sell_loss = percentages[0]
        
        if std > 2:
            std = 2
        
        if std > 1:
            
            sell_limit = std - std*0.1
            sell_loss = - std*0.75
        
        stop_loss  = max_price * (100 + sell_loss) / 100
        
        for new_i in range(1, future_prediction_range):
            
            new_price = df.iloc[i + new_i]['close']
            low = df.iloc[i + new_i]['low']
            
            gain = 100 * (new_price - current_price + current_price*1e-7) / current_price
            
            
            if new_price > max_price:
                max_price = new_price
                stop_loss  = max_price * (100 + sell_loss) / 100
                
            if gain > max_gain:
                max_gain = gain
                
            if gain < max_drawdown:
                max_drawdown = gain
                
            if max_gain >= sell_limit and max_drawdown < -0.05:
                label = labelling[1]
                break
                
            elif max_gain >= sell_limit and df.iloc[i + 1]['close'] - current_price > 0:
                label = labelling[-1]
                #print(sell_limit, max_gain, current_price, new_price)
                break
                
            #elif new_price < stop_loss and max_gain >= 0.4:
             #   label = 1
               # break
            
            elif new_price < stop_loss and max_gain <= 0.05 and df.iloc[i + 1]['close'] - current_price < 0:
                label = labelling[0]
                #print(sell_limit, max_gain, current_price, new_price, stop_loss, max_drawdown, sell_loss,max_price)
                break
            
            elif new_price < stop_loss and max_gain > 0.01:
                label = labelling[1]
                break
            
            elif new_i == (future_prediction_range - 1):
                label = labelling[1]
                
            else:
                label = labelling[1]
                
            previous_price = new_price
            
        ML_values.append(label)
        std_values.append(std)
        
    for _ in range(future_prediction_range):
        ML_values.append(np.nan)
        std_values.append(np.nan)
        
    df['labels'] = ML_values
    df['std'] = std_values
    
    return df
    
    
def paper_trades(df, graph):
    active = False
    money = 100
    percentage = 0
    for i, value in enumerate(df['labels']):
        if active == False and value == 1:
            print("Buy!")
            money = money - money*0.001
            buy_price = df.iloc[i]['close']
            graph.buy(df.iloc[i]['time'], buy_price)
            active = True
            
        if active == True and value == -1:
            print("Sell!")
            money = money - money*0.001
            sell_price =df.iloc[i]['close']
            graph.sell(df.iloc[i]['time'], sell_price)
            active = False
            percentage_diff = 100*(sell_price - buy_price) / buy_price
            money = money + money*(percentage_diff / 100)
            print("percentage_diff", percentage_diff)
            percentage += percentage_diff

    print(f"Percentage gain = {round(percentage, 1)}%")
    print(f"Money {money}£")
    
    return money


if __name__ == "__main__":
    
    from GUI import GraphApp
    
    #"AAVEUSDT"
    symbol = "AAVEUSDT"
    data = get_data_binance(symbol = symbol, interval='1d',  start = "1 Jun,2020", end = "1 Jun,2022", save = None)
    #data = load_data(symbol = "BTCUSDT")
    
    """
    #data = create_labels(data).iloc[:-20]
    data = create_labels_slow(data).iloc[:-20]
    
    graph = GraphApp(data)
    graph.indicator_subchart('labels')
    graph.indicator_subchart('labels')
    graph.show()
    paper_trades(data, graph)
    """
