from tradelog import trade_log_df   # Your trade log DataFrame
print(trade_log_df.columns)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
import torch

# Set global seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

def safe_float(x, default=0.0):
    """Converts x to float, returns default if x is None or NaN."""
    try:
        val = float(x)
        if np.isnan(val):
            return default
        else:
            return val
    except Exception:
        return default

class TradingEnv(gym.Env):
    """
    A custom Trading Environment for Deep Q-Learning where the agent learns whether to 
    enter a trade when a trade signal is available. Each trade opportunity is one step.
    
    Action space:
       0 = Skip trade,
       1 = Enter trade.
    Reward:
       Equal to the realized PnL of the trade if taken, otherwise 0.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, trade_log_df, initial_capital=10000000):
        super(TradingEnv, self).__init__()
        self.trade_log = trade_log_df.reset_index(drop=True)  # one row per trade opportunity
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.current_trade = 0

        # Binary action space: 0 = Skip, 1 = Enter
        self.action_space = spaces.Discrete(2)
        
        # Observation space: 5-dimensional state vector
        # [Entry Z-Score, Entry Normalized Elo Diff, Elo Change at Entry, 
        #  Market Turbulence at Entry, Position Multiplier]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def _get_observation(self):
        if self.current_trade < len(self.trade_log):
            row = self.trade_log.iloc[self.current_trade]
            obs = np.array([
                safe_float(row.get("Entry Z-Score", 0), default=0.0),
                safe_float(row.get("Entry Normalized Elo Diff", 0), default=0.0),
                safe_float(row.get("Elo Change at Entry", 0), default=0.0),
                safe_float(row.get("Market Turbulence at Entry", 0), default=0.0),
                safe_float(row.get("Position Multiplier", 1), default=1.0)
            ], dtype=np.float32)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        self.current_trade = 0
        self.capital = self.initial_capital
        return self._get_observation(), {}

    def step(self, action):
        if self.current_trade >= len(self.trade_log):
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0, True, False, {}
        
        current_trade_row = self.trade_log.iloc[self.current_trade]
        if action == 1:  # Enter trade
            reward = current_trade_row["PnL"]
            self.capital += reward
        else:            # Skip trade
            reward = 0
        
        self.current_trade += 1
        obs = self._get_observation()
        terminated = self.current_trade >= len(self.trade_log)
        return obs, reward, terminated, False, {}

    def render(self, mode="human"):
        print(f"Trade: {self.current_trade} | Capital: {self.capital}")

# ---------------------------
# Main training and testing code
# ---------------------------
if __name__ == "__main__":
    # Load your trade log DataFrame (ensure it has the required columns).
    # Here, we assume trade_log_df is provided by your module.
    from tradelog import trade_log_df
    print("Trade Log Columns:", trade_log_df.columns)

    # Create the custom trading environment.
    env = TradingEnv(trade_log_df)
    
    # Optional: check environment compliance with gymnasium.
    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)
    
    # Import the DQN algorithm from stable-baselines3.
    from stable_baselines3 import DQN
    
    # Instantiate the DQN agent with an MLP policy.
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, seed=seed_value)
    
    # Train the agent. Adjust total_timesteps as needed.
    model.learn(total_timesteps=2000000)
    
    # Save the trained model.
    model.save("dqn_trading_model")
    
    # Testing: reset environment and run episode using the trained policy.
    obs, _ = env.reset()
    terminated = False
    total_reward = 0
    while not terminated:
        # Use the trained model's policy.
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
    
    print("Test Total Reward:", total_reward)
