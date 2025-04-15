from tradelog import trade_log_df


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    A custom Trading Environment for Deep Q-Learning which lets the agent decide whether to 
    enter a trade when the trade signal is available. Each trade opportunity is taken 
    as one step. The action space is binary:
      0 = Skip trade
      1 = Enter trade
    The reward is the PnL of the trade if entered, and 0 if skipped.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, trade_log_df, initial_capital=10000000):
        super(TradingEnv, self).__init__()
        self.trade_log = trade_log_df.reset_index(drop=True)  # trade log DataFrame, one row per trade opportunity
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.current_trade = 0

        # Binary action space: 0 = Skip, 1 = Enter trade
        self.action_space = spaces.Discrete(2)
        
        # Define observation space: we'll use a 5-dimensional state extracted from the trade log.
        # You can adjust the number of dimensions and bounds based on your features.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def _get_observation(self):
        """
        Returns the observation vector for the current trade opportunity.
        For example, we extract key features from the current row:
            [Entry Z-Score, Entry Normalized Elo Diff, Elo Change at Entry, 
             Market Turbulence at Entry, Position Multiplier].
        If there are no more trades, return zeros.
        """
        if self.current_trade < len(self.trade_log):
            row = self.trade_log.iloc[self.current_trade]
            obs = np.array([
                row.get("Entry Z-Score", 0),
                row.get("Entry Normalized Elo Diff", 0),
                row.get("Elo Change at Entry", 0),
                row.get("Market Turbulence at Entry", 0),
                row.get("Position Multiplier", 1)
            ], dtype=np.float32)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        """
        Reset the environment state.
        Returns: (observation, info)
        """
        self.current_trade = 0
        self.capital = self.initial_capital
        return self._get_observation(), {}

    def step(self, action):
        """
        Takes an action and returns (observation, reward, terminated, truncated, info).
        The action space:
            0: Skip trade (reward=0)
            1: Enter trade (reward = trade PnL from the current trade row)
        After an action, the environment moves to the next trade opportunity.
        """
        if self.current_trade >= len(self.trade_log):
            # No more trades: terminate the episode
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0, True, False, {}

        # Get the current trade opportunity
        current_trade_row = self.trade_log.iloc[self.current_trade]
        
        if action == 1:  # Enter trade
            reward = current_trade_row["PnL"]
            self.capital += reward
        else:            # Skip trade
            reward = 0
            
        self.current_trade += 1  # Move to next trade opportunity
        
        obs = self._get_observation()
        terminated = self.current_trade >= len(self.trade_log)
        return obs, reward, terminated, False, {}

    def render(self, mode="human"):
        print(f"Current Trade: {self.current_trade} | Capital: {self.capital}")

# Example usage of the environment with a random agent
if __name__ == "__main__":
    env = TradingEnv(trade_log_df)
    
    obs, _ = env.reset()
    terminated = False
    total_reward = 0
    
    while not terminated:
        # Instead of using random actions, we use a heuristic:
        # Check the current trade row from the trade log.
        if env.current_trade < len(env.trade_log):
            current_trade_row = env.trade_log.iloc[env.current_trade]
            # If the expected trade PnL is positive, choose to enter (action 1), else skip (action 0)
            action = 1 if current_trade_row["PnL"] > 0 else 0
        else:
            action = 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        
    print("Total Reward:", total_reward)

