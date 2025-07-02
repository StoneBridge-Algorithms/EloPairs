# EloPairs

EloPairs is an experimental pairs trading project that combines traditional spread trading rules with an **Elo rating** signal and reinforcement learning (RL). The repository demonstrates how to engineer features, generate trade logs, and train RL agents to decide whether to enter a given trade.

## Feature Engineering

1. **Pair Spreads**  
   For symbols $A$ and $B$ the log prices are $\log P_A$ and $\log P_B$. We estimate a hedge ratio $\beta$ using OLS:
   \[
   \log P_A(t) = \alpha + \beta \cdot \log P_B(t) + \varepsilon_t.
   \]
   The spread is then
   \[
   s_t = \log P_A(t) - \beta\,\log P_B(t).
   \]
   A rolling mean and standard deviation of $s_t$ yield the Z-score
   \[
   z_t = \frac{s_t - \mu_t}{\sigma_t}.
   \]
2. **Elo Ratings**  
   Each stock receives an Elo rating that updates according to relative returns $r_A$ and $r_B$ at time $t$:
   \[
   R_A^{\text{new}} = R_A + K \left(S_A - E_A\right),\qquad
   R_B^{\text{new}} = R_B + K \left(1-S_A - (1-E_A)\right),
   \]
   where $E_A = \frac{1}{1 + 10^{(R_B-R_A)/400}}$ and $S_A$ encodes which stock outperformed. The difference $R_A - R_B$ is used as an additional momentum signal.

The script `build_dataset.py` loads price data (`NSE_PFC_EQ_candlestick_data.csv` and `NSE_RECLTD_EQ_candlestick_data.csv`), computes these features, and writes `PFC_RECLTD_feature_set.csv`.

## Trading Logic

`tradelog.py` reads the engineered features and creates a trade log. Positions open when:

- $z_t \geq 1.5$ **and** the normalized Elo difference decreases by at least 0.38 → short the spread.
- $z_t \leq -1.5$ **and** the normalized Elo difference increases by at least 0.38 → long the spread.

Trades close when the opposite conditions occur. PnL is calculated with a position size that can scale down when a market turbulence index is high. The resulting trades are assembled into `trade_log_df` and an equity curve can be plotted.

## Reinforcement Learning

Two environments (`dqltradeLog.py` and `ppo_trading_agent.py`) expose each trade opportunity as a step with action space `{0=skip, 1=enter}`. Rewards correspond to realized trade PnL. The included examples train:

- A Deep Q-Network (DQN) agent (`trainingdwnonTradelog.py`).
- A Proximal Policy Optimization (PPO) agent (`ppo_trading_agent.py`).

Both rely on [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). Successful training requires the dependencies listed below.

## File Overview

- `build_dataset.py` – construct feature set with spreads, z-scores and Elo ratings.
- `turbulence_calc.py` – compute a simple turbulence index from index returns.
- `tradelog.py` – generate entry/exit signals and PnL, producing `trade_log_df`.
- `dqltradeLog.py` – minimal RL environment for DQN.
- `trainingdwnonTradelog.py` – trains the DQN agent.
- `ppo_trading_agent.py` – example PPO training script.
- `elodiff_graph.py` – plot z-scores and Elo differences.

CSV files under the repo provide sample candlestick data for the two stocks and the NIFTY index.

## Requirements

The project uses Python 3 with packages such as `pandas`, `numpy`, `gymnasium`, `stable-baselines3`, `statsmodels`, and `plotly`. Install via:

```bash
pip install pandas numpy gymnasium stable-baselines3 statsmodels plotly
```

## Quick Start

1. Run `build_dataset.py` to create the feature CSV.
2. Execute `tradelog.py` to produce the trade log and view an equity curve.
3. Optionally train a RL agent using `trainingdwnonTradelog.py` or `ppo_trading_agent.py`.

This repository serves as a sandbox for exploring Elo-based pairs trading and RL-driven trade selection. Extensive backtesting and risk management are needed before considering live deployment.

