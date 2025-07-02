# EloPairs Project Summary

**EloPairs** explores a hybrid pairs-trading strategy that augments classic spread reversion with an Elo-rating momentum signal. The repository contains utilities to build features, generate trade logs and train RL agents to select trades.

## Trading Setup
- **Spread and Z-Score**: Using OLS, we compute a hedge ratio \(\beta\) between PFC and RECLTD closing prices. The log-price spread \(s_t = \log PFC_t - \beta \log RECLTD_t\) is normalized to a z-score. Extremes of this z-score indicate potential mean reversion.
- **Elo Ratings**: Each stock receives a rating that updates after each day depending on which stock outperformed. The Elo difference acts as a momentum-style feature. Rapid drops/rises in the normalized difference help time entries.
- **Trade Rules**: `tradelog.py` opens a short (long) spread when the z-score is above 1.5 (below -1.5) and the Elo difference change crosses -0.38 (+0.38). Exits occur when the opposite conditions trigger. Market turbulence can scale position size.
- **Reinforcement Learning**: `dqltradeLog.py` and `ppo_trading_agent.py` provide RL environments. Each trade opportunity becomes a step where the agent chooses whether to act, using realized PnL as reward.

## Code Structure
- **Data Preparation** – `build_dataset.py` loads price CSVs, calculates spreads, z-scores and Elo features, and saves `PFC_RECLTD_feature_set.csv`.
- **Signal Generation** – `tradelog.py` merges a turbulence index and implements entry/exit logic, writing a trade log and building an equity curve.
- **RL Agents** – Training scripts leverage Stable-Baselines3 to learn when to take signals. Example PPO training is in `ppo_trading_agent.py`.
- **Analysis Utilities** – `opstra.py` contains statistical tests and plotting helpers. `elodiff_graph.py` visualizes Elo and spread behaviour.

## Scope and Future Work
This project is a prototype to examine whether Elo ratings add value to a standard pair trade. Open directions include:
- Parameterising thresholds and paths via configuration rather than hard-coding.
- Extending backtests with realistic transaction costs and walk-forward validation.
- Allowing the RL agent to size positions or manage exits.
- Comparing Elo signals with alternative momentum or sentiment measures.

## Literature Context
Pairs trading and z-score based entry rules are widely studied in quantitative finance (e.g., Gatev et al., 2006). Elo ratings originate from chess but have been applied to sport ranking and have niche usage in finance. Combining Elo ratings with RL-based trade selection is relatively unexplored, so systematic testing of this approach could yield novel insights.

## Conclusion
EloPairs demonstrates how classical statistical-arbitrage concepts can be combined with rating-based features and reinforcement learning. While initial results are promising on sample data, thorough evaluation and risk management are necessary before deployment in a live trading environment.
