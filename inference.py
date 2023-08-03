# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:39:10 2023

@author: Pedro
"""

import joblib
from tensorflow.keras import models
import matplotlib.pyplot as plt
import os
import numpy as np
from xgboost import XGBRegressor
from stable_baselines3 import DQN

from market_data import get_data_binance, load_data
from training_data import numerai_corr, create_dataset, create_dataset_extended, create_indicators, postprocessing, feature_names



def ml_inference(data, dataset_extended = False, model_name = "xgb_reg_v1.0", column_name = "xgb", window_size = 10):

    file = os.path.join("models", model_name)
    
    flatten = True
    
    if model_name[:7] == "tf_LSTM":
        flatten = False
        dataset_extended = True
    
    if model_name[:3] == "xgb":
        model = XGBRegressor()
        model.load_model(file +".json")
        
        sorted_idx = model.feature_importances_.argsort()
        plt.barh(np.array(feature_names)[sorted_idx], model.feature_importances_[sorted_idx])
        plt.show()
    
    elif model_name[:2] == "tf":
        model = models.load_model(file + ".h5")
        
    elif model_name == 'RL':
        model = DQN.load(file)

    scale_input = joblib.load(file + "_input.gz")
    #scale_output = joblib.load(file + "_output.gz")
    
    if dataset_extended == True:
        X, _ = create_dataset_extended(data, back_range = window_size, flatten = flatten)
        data = data.iloc[100 + window_size:-20]
        
    else:
        X, _ = create_dataset(data)
        data = data.iloc[100:-20]

    #X = data[['volume_train', 'high_diff', 'low_diff', 'MA30_diff', 'MA100_diff', 'rsi', 'macd']]
    
    if model_name[:7] == "tf_LSTM":
        scaledX = scale_input.transform(X.reshape(X.shape[0],-1)).reshape(X.shape)
        
    else:
        scaledX = scale_input.transform(X)
        
    if model_name == 'RL':
        preds = model.predict(scaledX, deterministic = True)[0]
        
    else:
        preds = model.predict(scaledX)
     
    #scaled_preds = postprocessing(preds, scale_output)

    data[column_name] = preds
    
    return data

def calculate_magnitude_percentages(arr, values):
    arr.sort()  # Sort the array in ascending order
    total_values = len(arr)
    percentages = []

    for value in values:
        if value in arr:
            index = arr.index(value)  # Get the index of the value in the sorted array
            percentage = (index / (total_values - 1)) * 100  # Calculate the percentage
            percentages.append(percentage)
        else:
            percentages.append(None)  # Value not found in the array, add None to the percentages list

    return percentages

def signal_to_percentage(data, signal):
    data[signal + '_percentages'] = calculate_magnitude_percentages(data[signal].tolist(), data[signal].tolist())
    
    return data
    

def signal_merger(data, signals):
    
    merged_signals = np.zeros(len(data))
    
    for signal in signals:
        data = signal_to_percentage(data, signal)
        merged_signals += data[signal + '_percentages']
        
    data['merged_signal'] = merged_signals / len(signals)
    
    return data


def proportion_of_succesfull_trades(percentage_diff_list, threeshold = 0.2):
    count = len([x for x in percentage_diff_list if x > threeshold])
    succesfull_trades = round((count / (len(percentage_diff_list)+1e-7)) * 100)
    fail_trades = 100 - succesfull_trades
    
    info_proportion = f"{succesfull_trades}% succesfull trades and {fail_trades}% lost trades, with threeshold {threeshold}%"

    return info_proportion

    
def paper_trades(df, column_name):
    active = False
    money = 100
    percentage = 0
    median = df[column_name].median()
    minn = df[column_name].min()
    maxx = df[column_name].max()
    mean = df[column_name].mean()
    std = df[column_name].std()
    
    buy_threeshold = median + 0.5*median
    sell_threeshold = (median + minn) / 2
    #buy_threeshold = mean + 1*std
    #sell_threeshold = (median + minn) / 2
    # 83 as the buy_threeshold works well
    buy_threeshold = 83
    # 20/30 as the sell_threeshold works well
    sell_threeshold = 20
    
    percentage_diff_list = []
    
    previous_value = 50
    
    
    for i, value in enumerate(df[column_name]):
        
        '''
        if active == False and df.iloc[i]['MA25-6_diff'] > -0 and df.iloc[i]['rsi'] < 90 and value < buy_threeshold and previous_value > buy_threeshold:
            print("Buy!")
            money = money - money*0.001
            buy_price = df.iloc[i]['close']
            graph.buy(df.iloc[i]['time'], buy_price)
            active = True
            
        if active == True and df.iloc[i]['MA25-6_diff'] < 0 and df.iloc[i]['rsi'] > 10 and value > sell_threeshold and previous_value < sell_threeshold:
            print("Sell!")
            money = money - money*0.001
            sell_price =df.iloc[i]['close']
            graph.sell(df.iloc[i]['time'], sell_price)
            active = False
            percentage_diff = 100*(sell_price - buy_price) / buy_price
            if percentage_diff < -0.5:
                percentage_diff = -0.5
            money = money + money*(percentage_diff / 100)
            print("percentage_diff", percentage_diff)
            percentage += percentage_diff
            percentage_diff_list.append(percentage_diff)
            '''
            
        current_price = df.iloc[i]['close']
        
        percentages = [-0.5, 0.7]
            
        std  = data.iloc[i]['std']
        
        sell_limit = percentages[-1]
        sell_loss = percentages[0]
        
        if std > 2:
            std = 2
            
        if std > 1:
            
            sell_limit = std - std*0.1
            sell_loss = - std*0.75
            
        sell_loss *= 0.5
        
        buy_threeshold = 70
        condition2 = data.iloc[i]['LSTM10_percentages'] > buy_threeshold
        #condition2 = True
        
        if active == False:
            if value == 2 and condition2:
                print("Buy!")
                money = money - money*0.001
                buy_price = df.iloc[i]['close']
                graph.buy(df.iloc[i]['time'], buy_price)
                active = True
                max_price = buy_price
                stop_loss  = buy_price * (100 + sell_loss) / 100
            
        elif active == True:
            
            percentage_diff = 100*(current_price - buy_price) / buy_price
            
            stop_loss  = max_price * (100 + sell_loss) / 100
            
            if current_price > max_price and percentage_diff > -sell_loss:
                max_price = current_price
                stop_loss  = max_price * (100 + sell_loss) / 100
              
            low = df.iloc[i]['low']  
            
            condition = current_price < stop_loss
            #condition = False
            
            if value == 0 or condition:
                print("Sell!")
                money = money - money*0.001
                sell_price =df.iloc[i]['close']
                graph.sell(df.iloc[i]['time'], sell_price)
                active = False
                percentage_diff = 100*(sell_price - buy_price) / buy_price
                if current_price < stop_loss:
                    percentage_diff = 100*(stop_loss - buy_price) / buy_price
                    pass
                if percentage_diff < -0.5:
                    #percentage_diff = -0.5
                    pass
                money = money + money*(percentage_diff / 100)
                print("percentage_diff", percentage_diff)
                percentage += percentage_diff
                percentage_diff_list.append(percentage_diff)
            
        previous_value = value
            
    print(proportion_of_succesfull_trades(percentage_diff_list, threeshold = 0.2))
    print(proportion_of_succesfull_trades(percentage_diff_list, threeshold = 0))
    change_in_price = round(100*((df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close']))
    print(f"Price change during test period {change_in_price}%")
    print(f"Percentage gain = {round(percentage, 1)}%")
    print(f"Money {money}£")
    
    return money

def LSTM_inference(data, window_size = 10): 
    model_name = f"tf_LSTM{window_size}"
    column_name = f"LSTM{window_size}"
    data = ml_inference(data, dataset_extended = True, model_name = model_name, column_name = column_name, window_size = window_size)
    data = signal_to_percentage(data, column_name)
    data[column_name + f'_percentages'] = (data[column_name + f'_percentages'] / 50) -1
    return data

if __name__ == "__main__":
    
    from GUI import GraphApp
    

    symbol = "DOTUSDT_extended"
    #data = get_data_binance(symbol = symbol, interval='15m', start = "1 July,2023", end = "7 July,2023", save = None)
    data = load_data(symbol).iloc[-20000:]
    #data = create_indicators(data)
    # model_name options "xgb_reg_v1.0" "tf_v1.0" "tf_LSTM"
    model_name = "xgb_reg_v1.0"
    column_name = "xgb"
    data = ml_inference(data, dataset_extended = False, model_name = model_name, column_name = column_name)
    model_name = "tf_v1.0"
    column_name = "nn"
    data = ml_inference(data, dataset_extended = False, model_name = model_name, column_name = column_name)
    model_name = "RL"
    column_name = "RL"
    data = ml_inference(data, dataset_extended = False, model_name = model_name, column_name = column_name)
    
    #model = DQN.load("dqn_cartpole")
    #data['RL'] = model.predict(np.array(data[feature_names]).astype(float), deterministic=True)[0]
    
    #df['labels'] = [random.random() for i in range(len(df['time']))]
    
    data = LSTM_inference(data, window_size = 10)
    data = LSTM_inference(data, window_size = 5)
    
    #data = signal_merger(data, ["LSTM", "xgb", 'nn'])
    
    graph = GraphApp(data)
    #graph.indicator('MA30', color = "#add8e6")
    #graph.indicator('MA100', color = "#ffffed")
    graph.indicator('MA6', color = "#add8e6")
    graph.indicator('MA25', color = "#ffffed")
    #graph.indicator_subchart('LSTM', subchart_number = 1, color = "green")
    graph.indicator_subchart('RL', subchart_number = 1, color = "blue")
    #graph.indicator_subchart('xgb_percentages', subchart_number = 1, color = "red")
    #graph.indicator_subchart('nn_percentages', subchart_number = 1, color = "blue")
    graph.indicator_subchart('LSTM5_percentages', subchart_number = 2, color = "green")
    graph.indicator_subchart('LSTM10_percentages', subchart_number = 2, color = "blue")
    #graph.indicator_subchart('merged_signal', subchart_number = 2, color = "yellow")
    #graph.buy(data['time'].iloc[-100], 40.3)
    #graph.sell(data['time'].iloc[-40], 54.4)
    #graph.ml_indicator()
    #graph.macd()
    #paper_trades(data, "LSTM_percentages")
    paper_trades(data, "RL")
    graph.show()
    #graph.live(symbol)


