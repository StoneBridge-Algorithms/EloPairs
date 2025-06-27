import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# The trade_log_df DataFrame is generated in tradelog.py
from tradelog import trade_log_df

class TradingEnv(gym.Env):
    """Simple environment where the agent decides to take or skip each trade."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, trade_log_df: pd.DataFrame, initial_capital: float = 10_000_000):
        super().__init__()
        self.trade_log = trade_log_df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.current_trade = 0

        # Action: 0 = Skip, 1 = Enter trade
        self.action_space = spaces.Discrete(2)

        # Observation: basic features from the trade log
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        if self.current_trade < len(self.trade_log):
            row = self.trade_log.iloc[self.current_trade]
            obs = np.array([
                row.get("Entry Z-Score", 0.0),
                row.get("Entry Normalized Elo Diff", 0.0),
                row.get("Elo Change at Entry", 0.0),
                row.get("Market Turbulence at Entry", 0.0),
                row.get("Position Multiplier", 1.0),
            ], dtype=np.float32)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_trade = 0
        self.capital = self.initial_capital
        return self._get_observation(), {}

    def step(self, action: int):
        if self.current_trade >= len(self.trade_log):
            return self._get_observation(), 0.0, True, False, {}

        trade = self.trade_log.iloc[self.current_trade]
        reward = 0.0
        if action == 1:
            reward = float(trade["PnL"])
            self.capital += reward

        self.current_trade += 1
        obs = self._get_observation()
        terminated = self.current_trade >= len(self.trade_log)
        return obs, reward, terminated, False, {}

    def render(self, mode="human"):
        print(f"Trade {self.current_trade} | Capital: {self.capital:.2f}")

if __name__ == "__main__":
    # Training with PPO if stable-baselines3 is available
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env

        env = TradingEnv(trade_log_df)
        check_env(env, warn=True)
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
        model.learn(total_timesteps=1_000_000)
        model.save("ppo_trading_model")

        # Evaluate the trained agent
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(int(action))
            total_reward += reward
        print("Evaluation reward:", total_reward)
    except Exception as e:
        print("Training could not be completed:", e)
