import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# 1. Load and Clean Data
def load_pair_data(file_a, file_b, symbol_a='PFC', symbol_b='RECLTD'):
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)
    
    df = pd.merge(df_a, df_b, on='timestamp', suffixes=(f'_{symbol_a}', f'_{symbol_b}'))
    df = df.sort_values('timestamp').dropna().reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('Asia/Kolkata')
    return df

# 2. Feature Engineering
def compute_spread_and_hedge_ratio(df, symbol_a, symbol_b):
    df[f'log_close_{symbol_a}'] = np.log(df[f'close_{symbol_a}'])
    df[f'log_close_{symbol_b}'] = np.log(df[f'close_{symbol_b}'])
    
    df[f'returns_{symbol_a}'] = df[f'log_close_{symbol_a}'].diff()
    df[f'returns_{symbol_b}'] = df[f'log_close_{symbol_b}'].diff()

    X = sm.add_constant(df[f'log_close_{symbol_b}'])
    model = sm.OLS(df[f'log_close_{symbol_a}'], X, missing='drop').fit()
    beta = model.params[f'log_close_{symbol_b}']
    
    df['spread'] = df[f'log_close_{symbol_a}'] - beta * df[f'log_close_{symbol_b}']
    return df, beta

# 3. Statistical Tests
def run_cointegration_test(series_a, series_b):
    score, pvalue, _ = coint(series_a, series_b)
    print(f"\nCointegration Test:\nTest Statistic: {score:.4f}, P-value: {pvalue:.4f}")
    return pvalue

def adf_test(series, title=''):
    print(f"\nADF Test for {title}:")
    result = adfuller(series.dropna())
    stat, p, crit = result[0], result[1], result[4]
    print(f"ADF Statistic: {stat:.4f}, p-value: {p:.4f}")
    for k, v in crit.items():
        print(f"  {k}: {v}")
    return p

def hurst_exponent(ts, lags=range(2, 100)):
    ts = ts.dropna()
    tau = [np.sqrt(np.std(np.subtract(ts[lag:].values, ts[:-lag].values))) for lag in lags]
    slope = np.polyfit(np.log10(list(lags)), np.log10(tau), 1)
    return slope[0]

# 4. Z-Score Computation
def compute_z_score(df, window=30):
    df['spread_mean'] = df['spread'].rolling(window).mean()
    df['spread_std'] = df['spread'].rolling(window).std()
    df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    return df

# 5. Plotting Functions
def plot_rolling_correlation(df, symbol_a, symbol_b, window=30):
    rolling_corr = df[f'returns_{symbol_a}'].rolling(window).corr(df[f'returns_{symbol_b}'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=rolling_corr, mode='lines', name='Rolling Correlation'))
    fig.update_layout(title='30-Day Rolling Correlation', xaxis_title='Date', yaxis_title='Correlation')
    fig.show()

def plot_z_score(df, start_date='2022-01-31'):
    df = df[df['timestamp'] >= start_date]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['z_score'], mode='lines', name='Z-Score'))
    fig.add_hline(y=0, line_dash='dash', line_color='red')
    fig.add_hline(y=1.5, line_dash='dash', line_color='green')
    fig.add_hline(y=-1.5, line_dash='dash', line_color='green')
    fig.update_layout(title='Z-Score of the Spread', xaxis_title='Date', yaxis_title='Z-Score')
    fig.show()

# --- Main Workflow ---
if __name__ == '__main__':
    symbol_a = 'PFC'
    symbol_b = 'RECLTD'

    df = load_pair_data("NSE_PFC_EQ_candlestick_data.csv", "NSE_RECLTD_EQ_candlestick_data.csv")
    df, beta = compute_spread_and_hedge_ratio(df, symbol_a, symbol_b)
    
    print(f"Hedge Ratio: {beta:.4f}")
    corr = df[f'returns_{symbol_a}'].corr(df[f'returns_{symbol_b}'])
    print(f"30-day Return Correlation: {corr:.4f}")
    
    p_cointegration = run_cointegration_test(df[f'close_{symbol_a}'], df[f'close_{symbol_b}'])
    hurst = hurst_exponent(df['spread'])
    print(f"\nHurst Exponent: {hurst:.4f}")
    
    adf_p = adf_test(df['spread'], 'Spread')
    
    df = compute_z_score(df)
    plot_rolling_correlation(df, symbol_a, symbol_b)
    plot_z_score(df)
