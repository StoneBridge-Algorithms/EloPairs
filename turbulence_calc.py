# turbulence_calc.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def compute_empirical_turbulence(returns_series, window=20):
    turbulence_values = []
    for t in range(len(returns_series)):
        if t < window:
            turbulence_values.append(np.nan)
        else:
            window_slice = returns_series.iloc[t-window:t]
            mu = window_slice.mean()
            var = window_slice.var()
            if var == 0:
                turbulence_values.append(0)
            else:
                turbulence_t = ((returns_series.iloc[t] - mu) ** 2) / var
                turbulence_values.append(turbulence_t)
    return pd.Series(turbulence_values, index=returns_series.index, name="turbulence")

def load_turbulence_data(file_path="NSE_NIFTY50_INDEX_candlestick_data.csv", window=20):
    # Load the dataset, parsing timestamp as datetime and sorting chronologically
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Compute returns from closing prices
    df["return"] = df["close"].pct_change()
    df = df.dropna().copy()
    
    # Compute the turbulence index and add as a column
    df["turbulence"] = compute_empirical_turbulence(df["return"], window=window)
    
    return df

def plot_turbulence(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["turbulence"],
        mode="lines",
        name="Turbulence",
        line=dict(color="firebrick", width=2)
    ))
    fig.update_layout(
        title="Turbulence Index Over Time",
        xaxis_title="Date",
        yaxis_title="Turbulence Index",
        template="plotly_white",
        width=1200,
        height=600
    )
    fig.show()

def main():
    df = load_turbulence_data()
    print(df[["timestamp", "close", "return", "turbulence"]].head(30))
    plot_turbulence(df)
    return df  # Return the dataframe if needed

if __name__ == "__main__":
    main()
