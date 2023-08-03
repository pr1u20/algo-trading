# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 08:32:08 2023

@author: Pedro
"""

from environment import CustomEnv
from training_data import create_indicators, normalizeX, save_data
from market_data import get_data_binance, load_data
from inference import LSTM_inference

from tensorflow.keras import layers, Input, models, optimizers
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory
import joblib
import os
from typing import Callable

from gymnasium import spaces

import torch as th
import torch.nn as nn

from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
   
    
def build_tf_LSTM(self):
    
    inputs = Input(shape=(self.x_train.shape[-2:]), name="1")
    x = layers.LSTM(units=32)(inputs)
    # if activation Sigmoid, shows only the moments when to buy
    output = layers.Dense(1, activation='sigmoid')(x)
    
    agent = models.Model(inputs = inputs, outputs = output)
    agent.summary()
    # Building the model to find the optimal strategy
    strategy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit = 100000, window_length = 1)
    dqn = DQNAgent(model = agent, nb_actions = 3,
                   memory = memory, nb_steps_warmup = 1000,
    target_model_update = 1e-2, policy = strategy)
    dqn.compile('adam', metrics =['mae'])
    
    return agent

def rl_DQN():
    env = CustomEnv(data)
    env.reset()
    
    num_actions = env.action_space.n
    
    '''
    inputs = Input(shape=env.observation_space.shape, name="2")
    x = layers.LSTM(units=32)(inputs)
    # if activation Sigmoid, shows only the moments when to buy
    output = layers.Dense(num_actions, activation='sigmoid')(x)
    
    agent = models.Model(inputs = inputs, outputs = output)
    agent.summary()
    '''
    
    
    agent = models.Sequential()
    agent.add(layers.Flatten(input_shape =(1,)+env.observation_space.shape))
    agent.add(layers.Dense(64))
    #agent.add(layers.Activation('relu'))
    agent.add(layers.Dense(num_actions))
    agent.add(layers.Activation('sigmoid'))
    agent.summary()
    
    
    # Building the model to find the optimal strategy
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit = 100000, window_length = 1)
    #policy = BoltzmannQPolicy()
    dqn = DQNAgent(model = agent, nb_actions = num_actions,
                   memory = memory, nb_steps_warmup = 1000,
    target_model_update = 1e-2, policy = policy)
    dqn.compile(optimizers.Adam(learning_rate = 1e-3), metrics =['mae'])
    
    
    history = dqn.fit(env, nb_steps = 10000, action_repetition= 2, nb_max_episode_steps = None, nb_max_start_steps = 50, start_step_policy = None, visualize = False, verbose = 2)
    
    # Testing the learning agent
    dqn.test(env, nb_episodes = 1, visualize = True)
    
    obs, info = env.reset()
    env.render()
    for _ in range(len(data)):
        action, _states = dqn.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env._get_info()
            break
            obs, info = env.reset()
        
    print(env._get_info())
    
    
class CustomPolicy(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 13):
        #super().__init__(features_dim, [64, 64], nn.ReLU)
        super().__init__(observation_space, features_dim)
        #self.features_dim = features_dim
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        #n_input_channels = observation_space.shape[0]
        self.lstm = nn.LSTM(input_size=features_dim, hidden_size=1, num_layers=1)
        self.seq = nn.Sequential(
            #nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2*1, features_dim),
            nn.ReLU(),
            nn.Flatten(),
        )
    
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        x, _ = self.lstm(x)
        x = self.seq(x)
        return x
    
    
class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: spaces.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)
    
def train_rl(data, timesteps, file = None, save = True, retrain = False, render = True):
    
    if retrain == True:
        scale_input = joblib.load(file + "_input.gz")
        model = DQN.load(file)
        
    else:
        scale_input = None
    
    env = CustomEnv(data, scale = scale_input)
    #check_env(env)
    env.reset()
    
    print("[INFO] Training RL model.")
    if retrain == False:
        # "MlpPolicy"
        policy_kwargs = dict(
            #activation_fn=th.nn.ReLU,
            features_extractor_class=CustomPolicy,
            #net_arch=[32, 32],
        )
        #"""
        model = DQN("MlpPolicy", 
                    env, 
                    learning_rate = 3e-4,
                    buffer_size=1000000,
                    learning_starts=50000,
                    exploration_fraction = 0.1,
                    exploration_initial_eps = 1.0,
                    exploration_final_eps = 0.05,
                    #policy_kwargs=policy_kwargs, 
                    verbose=2)
        #"""
        """
        model = PPO("MlpPolicy", 
                    env, 
                    learning_rate = 3e-4,
                    verbose=0)
        """
        
        """
        model = RecurrentPPO("MlpLstmPolicy",
                             env,
                             stats_window_size=1000,
                             verbose=0)
        """
        
        
        
    else:
        model.set_env(env)
        
    model.learn(total_timesteps=timesteps, log_interval=1, progress_bar=True, reset_num_timesteps=not retrain)
    
    if save == True:
        
        model.save(file)
        if retrain == False:
            joblib.dump(env.scale, file + "_input.gz")
        
    
    env.plot = False
    obs, info = env.reset()
    if render:
        env.render()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env._get_info()
            break
            obs, info = env.reset()
        
    print(env._get_info())
    env.plot_training()
    
    
        
    return model

def inference_rl(data, file, render = True):
    
    scale_input = joblib.load(file + "_input.gz")
    model = DQN.load(file)
    
    env = CustomEnv(data, scale = scale_input, plot = False)
    #check_env(env)
    env.reset()
    
    obs, info = env.reset()
    if render:
        env.render()
        
    for _ in range(len(data)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env._get_info()
            break
            obs, info = env.reset()
        
    env.plot_results(name = 'testing')
    print(env._get_info())

    
    
    
if __name__ == "__main__":
    
    
    print("[INFO] Loading data.")

    symbol = "BTCUSDT_LSTM_extended"
    data = load_data(symbol = symbol)[110:-20]
    #data = get_data_binance(symbol = symbol, interval='5m', start = "1 July,2023", end = "2 July,2023", save = None)

    #data = create_indicators(data).iloc[100:-20]
    
    print("[INFO] LSTM inference.")
    #data = LSTM_inference(data[:])
    #save_data(data, "BTCUSDT_LSTM_")
    
    file = os.path.join("models", "DQNTest")
    train_rl(data[250000:255000], 1000000, file = file, retrain = False, render = False)
    
    inference_rl(data[255000:260000], file, render = False)
    
    # W1 (1000) -----> 103£(100K) - _£ (1M)
    # W2 (1000) -----> 122£ (100K) - 238£ (1M)
    # W5 (1000) -----> 163£(100K) - 381£ (1M)
    
    
    
    
    
    
    
    
    
    