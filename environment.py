# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 20:15:43 2023

@author: Pedro
"""

import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import quantstats as qs
import pandas as pd
import webbrowser

from training_data import feature_names, create_indicators, normalizeX
from market_data import get_data_binance, load_data
from inference import proportion_of_succesfull_trades, LSTM_inference
from GUI import GraphApp


class CustomEnv(gym.Env):
    
    def __init__(self, data, scale = None, plot = True):
        
        
        self.data = data
        self.feature_names = feature_names.copy()

        self.scale, self.data[self.feature_names] = normalizeX(self.data[self.feature_names], scale_object = scale)
            
        #add the machine learning indicator to the features
        #self.feature_names.append('LSTM_percentages')
        self.episode_rewards = []
        self.episode_balances = []
        self.episode_trades = []
        self.episode_success = []
        self.data_size = len(data)
        self.initial_money = 100
        self.window_size = 1
        self.trading_fees = 0.1 #%
        self._current_tick = None
        self._end_tick = len(self.data) - 1
        self._done = None
        self.open_position = False
        # open_position_LSTM is same as open_position, but is unnaffected by stop_loss. This is used to not place consecutive loosing trades.
        self.open_position_LSTM = False
        self.window = None
        self.render_mode = "human"
        self.reward = 0
        self.percentage_change_list = []
        self.balance = self.initial_money
        self.number_of_trades = 0
        self.plot = plot
        self.stop_loss_active = False # False to not have a stop loss.
        # observation_to_array False, if you want observation to be pandas dataframe
        self.observation_to_array = True
        self.data['RL'] = np.zeros(len(self.data))
        self.data['balances'] = np.full(len(self.data), self.balance)
        self.data['Percentage Diff'] = np.zeros(len(self.data))
        self.data['On Position'] = np.full(len(self.data), -1)
        #To initialize with -1 ---> np.full(len(self.data), -1)
        self.data['Ticks Since Buy'] = np.full(len(self.data), -1)
        
        self.feature_names.append('Percentage Diff')
        self.feature_names.append('On Position')
        self.feature_names.append('Ticks Since Buy')
        self.feature_names.append('LSTM_percentages')
        #self.feature_names.append('LSTM10_percentages')
        #self.feature_names.append('LSTM5_percentages')
        #self.feature_names.append('LSTM2_percentages')
        
        num_features = len(self.feature_names)
        num_actions = 3 # buy, sell, hold
        
        # Define the action space and observation space
        self.actions = [0, 1, 2]
        self.action_space = spaces.Discrete(num_actions, start = self.actions[0])  # Define the action space according to your requirements

        if self.window_size > 1:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, num_features), dtype=np.float64)  # Define the observation space according to your requirements
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float64)
        
    
    def reset(self, seed = None):
        # Reset the environment and return the initial observation
        self.close()
        random.seed(seed)
        self._start_tick = random.randint(self.window_size, self.window_size + 10)
        self._current_tick = self._start_tick
        self.balance = self.initial_money
        self.open_position = False
        self.open_position_LSTM = False
        self.current_price = self._get_current_price()
        self.previous_price = self.current_price
        self.reward = 0
        self.percentage_change_list = []
        self.balances = []
        self._done = False
        self.ticks_since_buy = 0
        self.number_of_trades = 0
        self.data['RL'] = np.zeros(len(self.data))
        self.data['balances'] = np.full(len(self.data), self.balance)
        self.data['Percentage Diff'] = np.zeros(len(self.data))
        self.data['On Position'] = np.full(len(self.data), -1)
        self.data['Ticks Since Buy'] = np.full(len(self.data), -1)
        self.max_price = 0
        self.truncated = False
        
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _get_obs(self):
        
        first_index = self._current_tick - self.window_size + 1
        second_index = self._current_tick + 1
        
        if self.open_position == True:
            
            self.current_price = self._get_current_price()
            self.percentage_change = ((self.current_price - self.buy_price) / self.buy_price) * 100
            self.data['Percentage Diff'].iloc[self._current_tick] = self.percentage_change
            self.data['On Position'].iloc[self._current_tick] = 1
            
            ticks_signal = (self.ticks_since_buy / 20) - 1
            self.data['Ticks Since Buy'].iloc[self._current_tick] = ticks_signal
            
            self.ticks_since_buy += 1
            

        obs = self.data.iloc[first_index:second_index][self.feature_names]
        if self.observation_to_array:
            obs = np.array(obs)
            
            if self.window_size == 1:
                obs = obs[0]
        
        return obs
 
    def _get_info(self):
        
        info_proportion_0_threeshold = proportion_of_succesfull_trades(self.percentage_change_list, threeshold = 0)
        info_proportion_02_threeshold = proportion_of_succesfull_trades(self.percentage_change_list, threeshold = 0.2)
            
        
        profit = self.balance / self.initial_money
        
        return {'total rewards': self.reward, 'balance': self.balance, 'profit': profit, 'number of trades': self.number_of_trades, 'Additional info': [info_proportion_0_threeshold, info_proportion_02_threeshold]}
    
    def _get_current_price(self):
        return self.data.iloc[self._current_tick]['close']
    
    def _get_current_low(self):
        return self.data.iloc[self._current_tick]['low']
    
    def _get_current_time(self):
        return self.data.iloc[self._current_tick]['time']
    
    def _get_current_label(self):
        return self.data.iloc[self._current_tick]['labels']
    
    def _get_current_std(self):
        return self.data.iloc[self._current_tick]['std']
    
    def _get_current_percentage_diff(self):
        return self.data.iloc[self._current_tick]['Percentage Diff']
    
    def _get_instant_chage(self, past_tick = 0):
        
        close = self.data.iloc[self._current_tick - past_tick]['close']
        openn = self.data.iloc[self._current_tick - past_tick]['open']
        
        return 100*(close - openn) / close
        
    
    def _update_balances(self):
        self.data['balances'].iloc[self._current_tick] = self.balance
        
    
    def _on_buy(self):
        self.open_position = True
        self.open_position_LSTM = True
        self.buy_price = self.current_price
        self.max_price = self.buy_price
        self.balance = self.balance * (100 - self.trading_fees) / 100
        self.max_price = self.current_price
        
        label = self._get_current_label()
        
        reward = 0
        reward = label * 5
        
        if self.render_mode == "human" and self.window is not None:
            self.graph.buy(self._get_current_time(), self.buy_price)
            
        self.ticks_since_buy = 0
            
        return reward
    
    def _no_position(self, action):
        
        self.truncated = False
        
        if action == self.actions[2]:
            reward = self._on_buy()
            #reward = 0
            
        elif action == self.actions[1]:
            #reward = - self.instant_percentage_change * 2
            reward = 0
            pass
            
        elif action == self.actions[0]:
            reward = 0
            #self.truncated = True
            #reward = -10
            pass
        
        return reward
    
    def _on_sell(self, stop = False):
        # if stop = True: it is a stop loss sell and money is lost.
        self.open_position = False
        if stop == False:
            self.open_position_LSTM = False
            
        self.sell_price = self.current_price
        if stop:
            self.sell_price = self.stop_loss
        self.balance = self.balance + self.balance*((self.percentage_change)/ 100)
        self.balance = self.balance * (100 - self.trading_fees) / 100
        self.percentage_change_list.append(self.percentage_change)
        
        gain = (self.percentage_change - 2*self.trading_fees)
        if gain >= 1:
            # gain**3 works well
            reward = gain**3
            reward = gain*5
            
        elif 0 < gain < 1:
            # gain**3 works well
            reward = gain**3
            reward = gain*5
            
        elif -1 < gain <= 0:
            #gain*3 works well
            reward = gain*3
            reward = gain*5
            
        elif gain <= -1:
            #gain ** 5 works well
            reward = gain**5
            reward = gain*5
            
        
        
        label = self._get_current_label()
        #reward = -label * 10
        
        if self.render_mode == "human" and self.window is not None:
            self.graph.sell(self._get_current_time(), self.sell_price)
            
        self.ticks_since_buy = 0
        self.number_of_trades += 1
        
        return reward
    
    def _stop_loss(self):
        
        std = self._get_current_std()
        low = self._get_current_low()
        previous_change = self._get_instant_chage(1)
        
        sell_loss = 0.7
        
        if std > 2:
            std = 2
            
        if std > 0.7:
            sell_loss = std
            
        if (previous_change / 2) > sell_loss:
            #sell_loss = previous_change / 2
            pass
            
        #sell_loss = 5
        
        self.stop_loss  = self.max_price * (100 - sell_loss) / 100
        
        reward = 0
        if low < self.stop_loss:
            self.percentage_change = ((self.stop_loss - self.buy_price) / self.buy_price) * 100
            self.sell_price = self.stop_loss
            reward = self._on_sell(stop = True)
            
        # Only activate the stop loss for taking profit when the percentage increase in price is higher than the sell_loss percentage
        win_loss_activation = self.percentage_change > sell_loss  
        #win_loss_activation = True
        
        if self.current_price > self.max_price and win_loss_activation:
            # Comment the next line to mantian the stop loss constant since the beginning. If uncommented stop loss keeps increasing with increase in price.
            self.max_price = self.current_price
            pass
            
        return reward
    
    def _open_position(self, action):
        
        self.percentage_change = ((self.current_price - self.buy_price) / self.buy_price) * 100
        #self.instant_percentage_change = ((self.current_price - self.previous_price) / self.previous_price) * 100
        self.truncated = False
        
        if action == self.actions[2]:
            reward = 0
            #self.truncated = True
            #reward = -10
            if self.instant_percentage_change < 0:    
                reward = self.instant_percentage_change * 5
                pass

        elif action == self.actions[1]:
            reward = 0
            if self.instant_percentage_change < 0:    
                reward = self.instant_percentage_change * 5
                pass
            
        elif action == self.actions[0]:
            reward = self._on_sell()
            #self.open_position_LSTM = False
            
        if self.stop_loss_active and self.open_position == True:
            reward = self._stop_loss()
            
        if self.ticks_since_buy > 20:
            #reward -= 10
            pass
        
        return reward
    
    def step(self, action):
        
        self.data.iloc[self._current_tick, self.data.columns.get_loc('RL')] = action
        
        self.current_price = self._get_current_price()
        self.instant_percentage_change = ((self.current_price - self.previous_price) / self.previous_price) * 100
            
        if self.open_position == False:
            
            reward = self._no_position(action)
            
            
        elif self.open_position == True:
            
            reward = self._open_position(action)
            
        
        self.reward += reward
        
        self.previous_price = self.current_price
        self._update_balances()
            
        if self._current_tick == self._end_tick:
            self._done = True
            observation = self._get_obs()
            info = self._get_info()
            print(info)
            
            self.episode_rewards.append(info['total rewards'])
            self.episode_balances.append(info['balance'])
            self.episode_trades.append(info['number of trades'])
            
            first_numbers = info['Additional info'][1]
            if first_numbers[2] != "%":
                success = int(first_numbers[:1])
            else:
                success = int(first_numbers[:2])
            
            self.episode_success.append(success)
            
            open_tab = False
            if len(self.data) > 100000:
                open_tab = True
                
            if self.plot and self.number_of_trades > 0:
                self.plot_results(open_tab = open_tab)
        
        else:
            self._current_tick += 1
                
            observation = self._get_obs()
            info = self._get_info()
            
        if self.render_mode == "human" and self.window is not None:
            self._update_render()
        
        
        return observation, reward, self._done, self.truncated, info
    
    def render(self, mode = 'human'):
        if self.window is None and self.render_mode == "human":
            self.graph = GraphApp(self.data.iloc[self._current_tick:self._current_tick+2])
            self.graph.indicator('MA6', color = "#add8e6")
            self.graph.indicator('MA25', color = "#ffffed")
            #self.graph.indicator_subchart('labels', subchart_number = 1, color = "blue")
            self.graph.indicator_subchart('RL', subchart_number = 1, color = "green")
            self.graph.indicator_subchart('LSTM10_percentages', subchart_number = 1, color = "red")
            self.graph.indicator_subchart('LSTM5_percentages', subchart_number = 1, color = "yellow")
            self.graph.indicator_subchart('LSTM2_percentages', subchart_number = 1, color = "blue")
            self.graph.indicator_subchart('macd', subchart_number = 2, color = "blue")
            #self.graph.indicator_subchart('On Position', subchart_number = 2, color = "red")
            #self.graph.indicator_subchart('Percentage Diff', subchart_number = 2, color = "orange")
            #self.graph.indicator_subchart('Ticks Since Buy', subchart_number = 2, color = "yellow")
            self.graph.show()
            self.window = True
            print("---------------------")
            
    def _update_render(self):
        new_data = self.data.iloc[self._current_tick-1:self._current_tick]
        self.graph.chart.update(new_data[['open', 'high', 'low', 'close', 'time', 'volume']].squeeze())
        self.graph.update_all(new_data)
        
    def plot_training(self):
        
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title("Rewards")
        
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_balances)
        plt.title("Balances (£)")
        
        plt.subplot(2, 2, 3)
        plt.plot(self.episode_trades)
        plt.title("Number of trades")
        
        plt.subplot(2, 2, 4)
        plt.plot(self.episode_success)
        plt.title("Successful trades (%)")
        
        plt.show()
        
    
    def plot_results(self, name = "performance_report", open_tab = False):
        benchmark = self.data.iloc[self._start_tick:]['close'].copy()
        benchmark.index = pd.to_datetime(self.data.iloc[self._start_tick:]['time'])
        #benchmark.index.name = 'date'
        returns = self.data.iloc[self._start_tick:]['balances'].copy()
        returns.index = pd.to_datetime(self.data.iloc[self._start_tick:]['time'])
        #returns.index.name = 'date'
        
        qs.reports.html(returns, benchmark, output=f"{name}.html", periods_per_year = 365*24*12)
        if open_tab == True:
            url = f"file:///C:/Users/Dell%20Precision/OneDrive%20-%20University%20of%20Southampton/Documents/Projects/Algorithmic%20Trading/{name}.html"
            new = 2
            webbrowser.open(url,new=new)
        
    
    def close(self):
        if self.window is not None:
            self.graph.chart.exit()
            self.window = None
            
def strategy(observation):
    lstm2 = observation.iloc[-1]['LSTM2_percentages']
    lstm5 = observation.iloc[-1]['LSTM5_percentages']
    lstm10 = observation.iloc[-1]['LSTM10_percentages']
    mean_lstm = (lstm2+lstm5+lstm10) / 3
    macd_current = observation.iloc[-1]['macd']
    macd_previous = observation.iloc[-2]['macd']
    current_change = env._get_instant_chage(0)
    previous_change = env._get_instant_chage(1)
    
    
    threeshold_buy = 0.7
    threeshold_sell = -0.3
    
    #the current candlestick and the past candlestick must be positive
    condition_price_change = (current_change > 0 and previous_change > 0)
    #condition_price_change = True
    
    #the slope of the MACD signal should be positive
    condition_MACD_change = macd_previous < macd_current
    condition_MACD_change = True
    condition_MACD_low = macd_current > -1
    condition_MACD_low = True
    
    condition_lstm_buy = (lstm5 > threeshold_buy and lstm10 > threeshold_buy and lstm2 > threeshold_buy)
    condition_mean_buy = mean_lstm > threeshold_buy
    # Comment next line to buy when any of the indicators are under the threeshold, not the mean.
    condition_lstm_buy = condition_mean_buy
    
    condition_sell = lstm5 < threeshold_sell or lstm2 < threeshold_sell
    condition_mean_sell = mean_lstm < threeshold_sell
    # Comment next line to sell when any of the indicators are under the threeshold, not the mean.
    condition_sell = condition_mean_sell
    
    if not env.open_position and not env.open_position_LSTM and condition_lstm_buy and condition_MACD_change and condition_MACD_low and condition_price_change:
        action = 2
        
    elif condition_sell:
        action = 0
        
    else:
        action = 1
        
    if env.open_position_LSTM and not env.open_position and lstm5 < 0 or lstm2 < 0:
        env.open_position_LSTM = False
        
    return action
    
    
if __name__ == "__main__":
    
    symbol = "BTCUSDT_extended"
    #symbol = "AKROUSDT"
    data = load_data(symbol = symbol)[110:-20][-5000:]
    #data = get_data_binance(symbol = symbol, interval='5m', start = "31 July,2023", end = None, save = None)
    
    #data = create_indicators(data)
    
    data = LSTM_inference(data, 2)
    data = LSTM_inference(data, 5)
    data = LSTM_inference(data, 10)
    
    
    env = CustomEnv(data)
    env.observation_to_array = False
    obs, info = env.reset()
    env.render()
    
    
    done = False
    while not done:
        # Choose an action
        #action = env.action_space.sample()
        action = strategy(obs)
    
        # Perform the chosen action in the environment
        obs, reward, done, _, info = env.step(action)
        
        print("Action:", action)
        # Print the current observation, reward, and done flag
        #print('Observation:', observation)
        print('Reward:', reward)
        print('Done:', done)
        
        #sleep(0)
        
        
    #env.plot_results()
    # Close the environment
    #env.close()
        