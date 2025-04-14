import pandas as pd
import plotly.graph_objects as go

def load_and_filter_data(file_path="PFC_RECLTD_feature_set.csv"):
    """
    Loads the dataset from `file_path`, sorts it by timestamp,
    filters the data for timestamps on or after April 1, 2024 (Asia/Kolkata timezone),
    calculates the normalized elo_diff, and returns the filtered DataFrame.
    """
    # Load the dataset (parsing 'timestamp' as datetime) and ensure it's in chronological order
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp')

    # Localize the start_date to the same timezone as the timestamp column
    start_date = pd.to_datetime("2017-07-04").tz_localize("Asia/Kolkata")

    # Filter dataset for timestamps on or after April 1, 2024
    df_filtered = df[df['timestamp'] >= start_date].copy()

    return df_filtered

def main():
    # Load the filtered data using our reusable function
    df_filtered = load_and_filter_data()

    # Create a Plotly figure
    fig = go.Figure()

    # Plot normalized elo_diff
    fig.add_trace(go.Scatter(
        x=df_filtered['timestamp'], 
        y=df_filtered['elo_diff_normalize'],
        mode='lines',
        name='Normalized Elo Difference',
        line=dict(color='blue')
    ))

    # Plot z_score
    fig.add_trace(go.Scatter(
        x=df_filtered['timestamp'], 
        y=df_filtered['z_score'],
        mode='lines',
        name='Z-Score',
        line=dict(color='red')
    ))

    # Add dotted horizontal lines at ±1.5 levels
    fig.add_hline(y=1.5, line_dash="dot", line_color="black", 
                  annotation_text="1.5", annotation_position="top left")
    fig.add_hline(y=-1.5, line_dash="dot", line_color="black", 
                  annotation_text="-1.5", annotation_position="bottom left")

    # Customize the layout of the plot
    fig.update_layout(
        title='Normalized Elo Difference and Z-Score Over Time (After April 2024)',
        xaxis_title='Date',
        yaxis_title='Normalized Value',
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Show the interactive plot
    fig.show()

if __name__ == '__main__':
    main()
