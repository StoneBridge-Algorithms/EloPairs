import pandas as pd
from elomethod import compute_elo_series
from opstra import load_pair_data, compute_spread_and_hedge_ratio, compute_z_score  # adapt as per your function structure

# --- SETTINGS ---
symbol_A = 'PFC'
symbol_B = 'RECLTD'
file_A = "NSE_PFC_EQ_candlestick_data.csv"
file_B = "NSE_RECLTD_EQ_candlestick_data.csv"

# --- STEP 1: Load + Merge + Clean ---
df = load_pair_data(file_A, file_B, symbol_A, symbol_B)

# --- STEP 2: Compute Spread + Z-Score ---
df, hedge_ratio = compute_spread_and_hedge_ratio(df, symbol_A, symbol_B)
df = compute_z_score(df)

# --- STEP 3: Add Elo Ratings ---
df = compute_elo_series(df, f'returns_{symbol_A}', f'returns_{symbol_B}', K=20, epsilon=0.0005)

# Optional: Add momentum/smoothing
df['elo_momentum'] = df['elo_diff'].diff()
df['elo_smooth'] = df['elo_diff'].ewm(span=10).mean()
df['elo_diff_normalize'] = (df['elo_diff'] - df['elo_diff'].mean()) / df['elo_diff'].std()

# --- STEP 4: Final Feature Selection ---
features = [
    'timestamp',
    f'close_{symbol_A}', f'close_{symbol_B}',
    f'returns_{symbol_A}', f'returns_{symbol_B}',
    'spread', 'z_score',
    'elo_A', 'elo_B', 'elo_diff',
    'elo_momentum', 'elo_smooth', 'elo_diff_normalize'
]

df_final = df[features].dropna()

# --- STEP 5: Save Final Dataset ---
output_path = f"{symbol_A}_{symbol_B}_feature_set.csv"
df_final.to_csv(output_path, index=False)
print(f" Saved final dataset to {output_path}")

