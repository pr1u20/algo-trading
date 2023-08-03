# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:55:00 2023

@author: Pedro
"""

from generate_labels import create_labels, create_labels_slow, paper_trades
from market_data import get_data_binance, load_data

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

import os
from scipy import stats
import pandas as pd
import pandas_ta as ta
import numpy as np
from tensorflow.keras import layers, Input, models
from xgboost import XGBClassifier, XGBRegressor
from random import shuffle


feature_names = ['volume_train', 'high_diff', 'low_diff', 'MA25-6_diff', 'MA30_diff', 'MA100_diff', 'rsi', 'macd', 'std']

WINDOW_SIZE = 5


def numerai_corr(preds, target):
  # rank (keeping ties) then Gaussianize predictions to standardize prediction distributions
  ranked_preds = (preds.rank(method="average").values - 0.5) / preds.count()
  gauss_ranked_preds = stats.norm.ppf(ranked_preds)
  
  # make targets centered around 0
  centered_target = target - target.mean()
  
  # raise both preds and target to the power of 1.5 to accentuate the tails
  preds_p15 = np.sign(gauss_ranked_preds) * np.abs(gauss_ranked_preds) ** 1.5
  target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5
  
  # finally return the Pearson correlation
  return np.corrcoef(preds_p15, target_p15)[0, 1]

def create_indicators(data, get_labels = True):
    data = data.reset_index(drop=True)
    prediction_range = 20
    print('[INFO] Creating labels.')
    if get_labels:
        data = create_labels_slow(data, prediction_range)
    
    data['volume_train'] = (data['volume'] - data['volume'].mean()) / data['volume'].mean()
    period = 33
    print('[INFO] Creating MACD.')
    data['macd'] = data.ta.macd()['MACDh_12_26_9'] / data['close']
    period = 14 
    print('[INFO] Creating rsi.')
    data['rsi'] = data.ta.rsi()
    period = 6
    print('[INFO] Creating MA6.')
    data[f'MA{period}'] =  data['close'].rolling(window=period).mean()
    #data[f'MA{period}_diff'] = (data['close'] - data[f'MA{period}']) / data['close']
    period = 25
    print('[INFO] Creating MA25.')
    data[f'MA{period}'] =  data['close'].rolling(window=period).mean()
    data[f'MA25-6_diff'] = (data['MA25'] - data['MA6'])  / data['close']
    period = 30
    print('[INFO] Creating MA30.')
    data[f'MA{period}'] =  data['close'].rolling(window=period).mean()
    data[f'MA{period}_diff'] = (data['close'] - data[f'MA{period}']) / data['close']
    period = 100
    print('[INFO] Creating MA100.')
    data[f'MA{period}'] =  data['close'].rolling(window=period).mean()
    data[f'MA{period}_diff'] = (data['close'] - data[f'MA{period}']) / data['close']
    print('[INFO] Creating high difference.')
    data['high_diff'] = (data['high'] - data['close']) / data['close']
    print('[INFO] Creating low difference.')
    data['low_diff'] = (data['low'] - data['close']) / data['close']
    
    return data

def drop_repeating_labels(X, Y):
    
    positions_0 = np.where(Y==0)[0]
    shuffle(positions_0)
    
    num_0 = len(positions_0)
    num_1 = len(Y[Y == 1])
    num_m1 = len(Y[Y == -1])
    num_deletions = num_0 - max(num_1, num_m1)
    
    Y = np.delete(Y, positions_0[:num_deletions])
    X = np.delete(X, positions_0[:num_deletions], axis = 0)

    return X, Y

def create_dataset(data, ):
    X = data[feature_names].iloc[100:-20]
    
    Y = data['labels'].iloc[100:-20]
    
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def create_dataset_extended(data, back_range = 10, flatten = False, drop_labels = False):
    
    x = data[feature_names].iloc[100:-20]
    Y = data['labels'].iloc[100 + back_range:-20]
    X = []
    
    for i in range(back_range, len(x)):
        
        if flatten == False:
            X.append(x.iloc[i-back_range:i])
        
        else:
            X.append(np.array(x.iloc[-back_range + i:i]).flatten())
            
    X = np.array(X)
    Y = np.array(Y)
        
    return X, Y

def save_data(data, save_data = "BTCUSDT"):
    file = os.path.join("data", save_data +'_extended.csv')
    data.to_csv(file, index = False, header=True)
        

def XY_from_data(data, dataset_extended = False, flatten = True, save_data = None, drop_labels = True):
    
    if not save_data == None:
        data = create_indicators(data)
        save_data(data, save_data)
    
    if dataset_extended == True:
        X, Y = create_dataset_extended(data, back_range = WINDOW_SIZE, flatten = flatten)
        
    else:
        X, Y = create_dataset(data)
        
    if drop_labels:
        X, Y = drop_repeating_labels(X, Y)
    
    return X, Y

def normalizeX(X, flatten = True, scale_object = None):
    print('[INFO] Scaling.')
    if not scale_object == None:
        scale = scale_object
        
        if flatten == False:
        
            scaledX = scale.transform(X.reshape(X.shape[0],-1)).reshape(X.shape)
            
        else:
            scaledX = scale.transform(X)
        
    else:
        scale = StandardScaler(with_mean = True, with_std = True)
        #scale = MinMaxScaler()
        
        if flatten == False:
        
            scaledX = scale.fit_transform(X.reshape(X.shape[0],-1)).reshape(X.shape)
            
        else:
            scaledX = scale.fit_transform(X)
        
    return scale, scaledX
    
    
    
def split_and_normalize(X, Y, flatten = True):
    
    scale, scaledX = normalizeX(X, flatten)
        
    x_train, x_test, y_train, y_test = train_test_split(scaledX, Y, test_size=0.05, random_state=1)
    
    print(f"Training dataset size = {x_train.shape}")
    print(f"Testing dataset size = {x_test.shape}")
    
    return scale, (x_train, x_test, y_train, y_test)


def prepare_data(data, dataset_extended = False, flatten = True, save_data = None, drop_labels = True):
    
    X, Y = XY_from_data(data, dataset_extended = dataset_extended, flatten = flatten, save_data = save_data, drop_labels = drop_labels)
    
    scale, (x_train, x_test, y_train, y_test) = split_and_normalize(X, Y, flatten = flatten)
    
    return scale, (x_train, x_test, y_train, y_test)

def multiple_symbols_data_preparation(symbols, dataset_extended = False, flatten = True, save_indicators = False, drop_labels = False):
    
    X = []
    Y = []
    
    num_symbols = len(symbols)
    
    for symbol in symbols:
        
        print(f"[INFO] Loading {symbol} data.")
        data = load_data(symbol = symbol)
        
        if save_indicators:
            save_data = symbol
        else:
            save_data = None
            
        x, y = XY_from_data(data, dataset_extended = dataset_extended, flatten = flatten, save_data = save_data, drop_labels = drop_labels)
        X.append(x)
        Y.append(y)
        
    
    if num_symbols == 1:
        X = X[0]
        Y = Y[0]
        
    else:
        X = np.concatenate([X[i] for i in range(num_symbols)], axis=0)
        Y = np.concatenate([Y[i] for i in range(num_symbols)], axis=0)
    
    return X, Y


def postprocessing(predictions, scale_output):
    
    predictions = np.array(predictions).reshape(-1, 1)
    
    scaled_preds = scale_output.transform(predictions)
    
    return scaled_preds


def evaluate(model, x_test, y_test):
    
    predictions = model.predict(x_test)
    
    mse = mean_squared_error(predictions, y_test)
    corr = numerai_corr(pd.Series(predictions), y_test)
    print(f"mse = {mse}")
    print(f"Corr = {corr}")
    
    return predictions




class Train():
    
    def __init__(self, symbols, dataset_extended = False, flatten = True, save_indicators = None, drop_labels = False):
        
        X, Y = multiple_symbols_data_preparation(symbols, dataset_extended = dataset_extended, flatten = flatten, save_indicators = save_indicators, drop_labels = drop_labels)
        self.scale_input, (self.x_train, self.x_test, self.y_train, self.y_test) = split_and_normalize(X, Y, flatten = flatten)
    
    def prepare_output(self, model):
        
        train_predictions = model.predict(self.x_train)
        
        scale_output = MinMaxScaler()
        
        train_predictions = np.array(train_predictions).reshape(-1, 1)
        
        scale_output.fit_transform(train_predictions)
        
        predictions = model.predict(self.x_test)
        
        predictions = np.array(predictions).reshape(-1, 1)
        
        scaled_preds = scale_output.transform(predictions)
        
        predictions = np.array(predictions).reshape(1, -1)[0]
        
        mse = mean_squared_error(predictions, self.y_test)
        corr = numerai_corr(pd.Series(predictions), self.y_test)
        print(f"mse = {mse}")
        print(f"Corr = {corr}")
        
        return scale_output
    
    def build_tf_nn(self):
        
        inputs = Input(shape=(self.x_train.shape[1],), name="1")
        x = layers.Dense(units=64)(inputs)
        #x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(.1)(x)
        
        x = layers.Dense(units=32)(x)
        #x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(.1)(x)
        
        x = layers.Dense(8, activation='sigmoid')(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs = inputs, outputs = output)
        model.summary()
        model.compile(optimizer='adam', loss='MSE', metrics=['binary_crossentropy', 'accuracy'])
        
        return model
    
    def build_tf_LSTM(self):
        
        inputs = Input(shape=(self.x_train.shape[-2:]), name="1")
        x = layers.LSTM(units=32)(inputs)
        # if activation Sigmoid, shows only the moments when to buy
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs = inputs, outputs = output)
        model.summary()
        model.compile(optimizer='adam', loss='MSE', metrics=['binary_crossentropy', 'accuracy'])
        
        return model
        
    
    def regr_sklearn(self):
        
        model = LinearRegression()
        
        model = self.build_regr_sklearn()
        
        model.fit(self.x_train, self.y_train)
        
        scale_output = self.prepare_output(model)
        
        return model, scale_output
    
    def tf_nn(self, save = False):
        model = self.build_tf_nn()
        
        model.fit(self.x_train, self.y_train, batch_size = 64, epochs=2)
        
        scale_output = self.prepare_output(model)
        
        if save:
            file = os.path.join("models", "tf_v1.0")
            model.save(file + ".h5")
            
            joblib.dump(scale_output, file + "_output.gz")
            joblib.dump(self.scale_input, file + "_input.gz")
            
        return model, scale_output
    
    def xgb(self, save = False):
        model = XGBRegressor()
        
        model.fit(self.x_train, self.y_train)
        
        scale_output = self.prepare_output(model)
        
        if save:
            file = os.path.join("models", "xgb_reg_v1.0")
            
            model.save_model(file + ".json")
            joblib.dump(scale_output, file + "_output.gz")
            joblib.dump(self.scale_input, file + "_input.gz")
            
        return model, scale_output
    
    def tf_LSTM(self, save = False):
        model = self.build_tf_LSTM()
        
        scale = MinMaxScaler()
        scaledY = scale.fit_transform(self.y_train.reshape(-1, 1))
        
        model.fit(self.x_train, self.y_train, batch_size = 64, epochs=2)
        
        scale_output = self.prepare_output(model)
        
        if save:
            file = os.path.join("models", f"tf_LSTM{WINDOW_SIZE}")
            model.save(file + ".h5")
            
            joblib.dump(scale_output, file + "_output.gz")
            joblib.dump(self.scale_input, file + "_input.gz")
            
        return model, scale_output
    
    

if __name__ == "__main__":
    
    print('[INFO] Loading data.')
    #symbols = ["BTCUSDT", "DOTUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "AAVEUSDT"]
    symbols = ["BTCUSDT_extended", "DOTUSDT_extended", "XRPUSDT_extended"]
    
    
    # if using _extended save_data == None else put the name of the symbol
    # dataset_extended to process different window size than one
    # Most of the prediction y outputs are 0, so drop_labels gets rid of many zeros, to have similar number of -1, 0, 1
    train = Train(symbols, dataset_extended = True, flatten = False, save_indicators = False, drop_labels = True)
    
    print('[INFO] Training model.')
    #model, scale_output = train.tf_nn(save = True)
    
    model, scale_output = train.tf_LSTM(save = True)
    #model, scale_output = train.xgb(save = True)
    
    preds = model.predict(train.x_test)
    
    scaled_preds = postprocessing(preds, scale_output)
# =============================================================================
#     file = os.path.join("models", "tf_v1.0")
#     model = models.load_model(file + ".h5")
# 
#     scale_input = joblib.load(file + "_input.gz")
#     scale_output = joblib.load(file + "_output.gz")
# 
#     preds = model.predict(train.x_test)
#     
#     scaled_preds = postprocessing(preds, scale_output)
# =============================================================================
    