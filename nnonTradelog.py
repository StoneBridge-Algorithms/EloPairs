from tradelog import trade_log_df


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    A custom Trading Environment for Deep Q-Learning based on a trade log.
    The reward is based on the trade's PnL.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, trade_log_df, initial_capital=10000000):
        super(TradingEnv, self).__init__()
        self.trade_log = trade_log_df.reset_index(drop=True)
        self.current_trade = 0
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_open = False
        self.current_entry_trade = None
        self.done = False
        
        # Define action space: 0 = Hold, 1 = Enter trade, 2 = Exit trade.
        self.action_space = spaces.Discrete(3)
        
        # Define observation space: This is a dummy example (5-dimensional).
        # Adjust the dimensions and bounds as per your actual features.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
    
    def _get_observation(self):
        """
        Return an observation from the current trade log row.
        Here we use a simple 5-dimensional vector; update as needed.
        """
        if self.current_trade < len(self.trade_log):
            row = self.trade_log.iloc[self.current_trade]
            # Example features: [Entry Z-Score, Entry Normalized Elo Diff, Elo Change at Entry, 
            # Market Turbulence at Entry, Position Multiplier]
            obs = np.array([
                row.get('Entry Z-Score', 0),
                row.get('Entry Normalized Elo Diff', 0),
                row.get('Elo Change at Entry', 0),
                row.get('Market Turbulence at Entry', 0),
                row.get('Position Multiplier', 1),
            ], dtype=np.float32)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        """
        Reset the environment state and return an initial observation.
        """
        self.current_trade = 0
        self.capital = self.initial_capital
        self.position_open = False
        self.current_entry_trade = None
        self.done = False
        return self._get_observation(), {}  # Gymnasium reset returns (observation, info)

    def step(self, action):
        """
        Take an action:
          0: Hold, 1: Enter trade, 2: Exit trade
        Returns: observation, reward, terminated, truncated, info
        """
        reward = 0
        terminated = False
        truncated = False
        info = {}

        if self.current_trade >= len(self.trade_log):
            terminated = True
            return np.zeros(self.observation_space.shape, dtype=np.float32), reward, terminated, truncated, info

        current_trade_row = self.trade_log.iloc[self.current_trade]

        # If not in position: only option is to "enter" (action 1)
        if not self.position_open:
            if action == 1:
                self.position_open = True
                self.current_entry_trade = current_trade_row
                reward = 0  # No PnL at entry
            else:
                reward = 0
        else:
            # If in a position, valid exit is action 2.
            if action == 2:
                # Use the PnL from the trade as the reward.
                reward = current_trade_row['PnL']
                self.capital += reward
                self.position_open = False
                self.current_trade += 1  # move to next trade after exit
            else:
                reward = 0

        obs = self._get_observation()

        if self.current_trade >= len(self.trade_log):
            terminated = True

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Current Trade: {self.current_trade}, Capital: {self.capital}")

if __name__ == "__main__":
    # For testing purposes, suppose we have trade_log_df loaded from a CSV
    env = TradingEnv(trade_log_df)
    
    # Reset environment
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        # For testing, choose a random action.
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()
    print("Total Reward:", total_reward)
