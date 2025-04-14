from elodiff_graph import load_and_filter_data
import pandas as pd
import plotly.graph_objects as go

# Invested capital per trade (for full exposure in each trade)
invested_capital = 10000000  # 10,000,000
beta = 1  # For simplicity, beta is set to 1

# Get the filtered DataFrame
df_filtered = load_and_filter_data()

# Display a sample of the data and the columns for verification
print(df_filtered.head())
print(df_filtered.columns)

# Reset the index so that it becomes 0, 1, 2, …
df_filtered = df_filtered.reset_index(drop=True)

# tradelog.py
from turbulence_calc import load_turbulence_data
import pandas as pd

# Use the function to load the turbulence DataFrame
turbulence_df = load_turbulence_data()
print(turbulence_df.head())

turbulence_df['timestamp'] = pd.to_datetime(turbulence_df['timestamp']).dt.tz_localize('Asia/Kolkata')

# Merge on the 'timestamp' column
df_filtered = df_filtered.merge(turbulence_df[['timestamp', 'turbulence']], on='timestamp', how='left')

trades = []
position_open = False
entry_details = {}
trade_type = None  # To indicate whether the active trade is 'short_spread' or 'long_spread'

for i in range(1, len(df_filtered)):
    current_z = df_filtered.loc[i, 'z_score']
    date = df_filtered.loc[i, 'timestamp']
    
    prev_norm_elo = df_filtered.loc[i-1, 'elo_diff_normalize']
    current_norm_elo = df_filtered.loc[i, 'elo_diff_normalize']
    elo_change = current_norm_elo - prev_norm_elo
    
    # Get current market turbulence from the filtered DataFrame
    current_turbulence = df_filtered.loc[i, 'turbulence']
    
    # Determine position multiplier if you want to adjust size by turbulence (example below)
    turbulence_threshold = 1.0
    if current_turbulence < turbulence_threshold:
        multiplier = 1.0  # full exposure
    else:
        multiplier = turbulence_threshold / current_turbulence  # reduced exposure
    
    if not position_open:
        # ENTRY condition for a short spread trade:
        # When z_score >= 1.5 and the normalized Elo difference drops (change <= -0.38)
        if current_z >= 1.5 and elo_change <= -0.38:
            position_open = True
            trade_type = 'short_spread'
            entry_pfc = df_filtered.loc[i, 'close_PFC']
            entry_recltd = df_filtered.loc[i, 'close_RECLTD']
            entry_details = {
                'Trade Type': trade_type,
                'Entry Date': date,
                'Entry Z-Score': current_z,
                'Entry Normalized Elo Diff': current_norm_elo,
                'Elo Change at Entry': elo_change,
                'Entry PFC Price': entry_pfc,
                'Entry RECLTD Price': entry_recltd,
                'Market Turbulence at Entry': current_turbulence,
                'Position Multiplier': multiplier
            }
        # ENTRY condition for a long spread trade:
        # When z_score <= -1.5 and the normalized Elo difference rises (change >= 0.38)
        elif current_z <= -1.5 and elo_change >= 0.38:
            position_open = True
            trade_type = 'long_spread'
            entry_pfc = df_filtered.loc[i, 'close_PFC']
            entry_recltd = df_filtered.loc[i, 'close_RECLTD']
            entry_details = {
                'Trade Type': trade_type,
                'Entry Date': date,
                'Entry Z-Score': current_z,
                'Entry Normalized Elo Diff': current_norm_elo,
                'Elo Change at Entry': elo_change,
                'Entry PFC Price': entry_pfc,
                'Entry RECLTD Price': entry_recltd,
                'Market Turbulence at Entry': current_turbulence,
                'Position Multiplier': multiplier
            }
    else:
        if trade_type == 'short_spread':
            # EXIT condition for short spread trade:
            # When z_score <= -1.5 and the normalized Elo difference reverses (change >= 0.38)
            if current_z <= -1.5 and elo_change >= 0.38:
                exit_pfc = df_filtered.loc[i, 'close_PFC']
                exit_recltd = df_filtered.loc[i, 'close_RECLTD']
                exit_details = {
                    'Exit Date': date,
                    'Exit Z-Score': current_z,
                    'Exit Normalized Elo Diff': current_norm_elo,
                    'Elo Change at Exit': elo_change,
                    'Exit PFC Price': exit_pfc,
                    'Exit RECLTD Price': exit_recltd,
                    'Market Turbulence at Exit': current_turbulence
                }
                # Calculate number of shares using position sizing with the multiplier:
                shares = multiplier * (invested_capital / (entry_details['Entry PFC Price'] + entry_details['Entry RECLTD Price']))
                # PnL for short spread:
                pnl = shares * ((entry_details['Entry PFC Price'] - exit_pfc) + beta * (exit_recltd - entry_details['Entry RECLTD Price']))
                trade = {**entry_details, **exit_details, 'PnL': pnl, 'Shares': shares}
                trades.append(trade)
                position_open = False
                entry_details = {}
                trade_type = None
        elif trade_type == 'long_spread':
            # EXIT condition for long spread trade:
            # When z_score >= 1.5 and the normalized Elo difference reverses (change <= -0.38)
            if current_z >= 1.5 and elo_change <= -0.38:
                exit_pfc = df_filtered.loc[i, 'close_PFC']
                exit_recltd = df_filtered.loc[i, 'close_RECLTD']
                exit_details = {
                    'Exit Date': date,
                    'Exit Z-Score': current_z,
                    'Exit Normalized Elo Diff': current_norm_elo,
                    'Elo Change at Exit': elo_change,
                    'Exit PFC Price': exit_pfc,
                    'Exit RECLTD Price': exit_recltd,
                    'Market Turbulence at Exit': current_turbulence
                }
                shares = multiplier * (invested_capital / (entry_details['Entry PFC Price'] + entry_details['Entry RECLTD Price']))
                # PnL for long spread:
                pnl = shares * ((exit_pfc - entry_details['Entry PFC Price']) + beta * (entry_details['Entry RECLTD Price'] - exit_recltd))
                trade = {**entry_details, **exit_details, 'PnL': pnl, 'Shares': shares}
                trades.append(trade)
                position_open = False
                entry_details = {}
                trade_type = None

# Convert the trade log list into a DataFrame for analysis
trade_log_df = pd.DataFrame(trades)
print(trade_log_df)

def build_equity_curve(trade_log):
    """
    Builds an equity curve by calculating cumulative PnL from the trade log.
    The curve is built over a daily time series from the first trade entry to the last trade exit.
    """
    # Sort the trade log by exit date to order trades chronologically.
    trade_log = trade_log.sort_values('Exit Date')
    
    # Compute cumulative PnL using exit dates
    cumulative_pnl_series = trade_log.set_index('Exit Date')['PnL'].cumsum()

    # Create a daily index from the first Entry Date to the last Exit Date
    start_date = trade_log['Entry Date'].min()
    end_date = trade_log['Exit Date'].max()
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Build a DataFrame for the equity curve and forward-fill the cumulative PnL
    equity_curve = pd.DataFrame(index=daily_dates, columns=['Cumulative PnL'])
    equity_curve['Cumulative PnL'] = cumulative_pnl_series.reindex(daily_dates, method='ffill').fillna(0)
    
    return equity_curve

def plot_equity_curve(equity_curve):
    """
    Creates an interactive Plotly chart for the equity curve.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve['Cumulative PnL'],
        mode='lines',
        name='Cumulative PnL',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title='Cumulative PnL Equity Curve',
        xaxis_title='Date',
        yaxis_title='Cumulative PnL',
        template='plotly_white'
    )
    
    fig.show()

def plot_equity_and_turbulence(equity_curve, df_filtered):
    """
    Overlays Cumulative PnL and Turbulence on a single plot 
    with two y-axes (left = PnL, right = Turbulence).
    """
    fig = go.Figure()
    
    # 1) Plot your cumulative PnL (y-axis 1)
    fig.add_trace(go.Scatter(
        x=equity_curve.index, 
        y=equity_curve['Cumulative PnL'],
        mode='lines',
        name='Cumulative PnL',
        line=dict(color='green', width=2),
        yaxis='y1'
    ))
    
    # 2) Plot the turbulence index (y-axis 2)
    fig.add_trace(go.Scatter(
        x=df_filtered['timestamp'],
        y=df_filtered['turbulence'],
        mode='lines',
        name='Turbulence',
        line=dict(color='firebrick', width=2),
        yaxis='y2'
    ))
    
    # 3) Configure layout for two y-axes
    fig.update_layout(
        title="Cumulative PnL & Market Turbulence Over Time",
        template="plotly_white",
        width=1200,
        height=600,
        xaxis=dict(
            domain=[0.0, 1.0], 
            title='Date'
        ),
        yaxis=dict(
            title='Cumulative PnL', 
            side='left'
        ),
        yaxis2=dict(
            title='Turbulence Index',
            side='right',
            overlaying='y'   # overlay on same x-axis
        )
    )
    
    fig.show()

if __name__ == '__main__':
    # Build the equity curve from the trade log data
    equity_curve = build_equity_curve(trade_log_df)
    print(equity_curve.head())
    
    # Calculate percentage return based on invested capital per trade
    final_cumulative_pnl = equity_curve['Cumulative PnL'].iloc[-1]
    final_equity = invested_capital + final_cumulative_pnl
    percentage_return = (final_equity - invested_capital) / invested_capital * 100
    print("Invested Capital per Trade: ", invested_capital)
    print("Final Equity Value: ", final_equity)
    print("Percentage Return: {:.2f}%".format(percentage_return))
    
    # Plot the equity curve
    plot_equity_curve(equity_curve)
    plot_equity_and_turbulence(equity_curve, df_filtered)
