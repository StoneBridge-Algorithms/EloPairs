import pandas as pd
import matplotlib.pyplot as plt

# Set the path to your final dataset CSV file
file_path = "PFC_RECLTD_feature_set.csv"

# Load the dataset (parsing 'timestamp' as datetime)
df = pd.read_csv(file_path, parse_dates=['timestamp'])
df = df.sort_values('timestamp')  # Ensure data is in chronological order

# Localize start_date to the same timezone as the timestamp column
start_date = pd.to_datetime("2024-04-01").tz_localize("Asia/Kolkata")

# Filter dataset for timestamps on or after April 1, 2024
df_filtered = df[df['timestamp'] >= start_date]

# Calculate normalized elo_diff: (value - mean) / std for the filtered data
df_filtered['normalized_elo_diff'] = (df_filtered['elo_diff'] - df_filtered['elo_diff'].mean()) / df_filtered['elo_diff'].std()

# Create the plot
plt.figure(figsize=(12, 6))

# Plot normalized elo_diff
plt.plot(df_filtered['timestamp'], df_filtered['normalized_elo_diff'], label='Normalized Elo Difference', color='blue')

# Plot z_score
plt.plot(df_filtered['timestamp'], df_filtered['z_score'], label='Z-Score', color='red')

# Customize the plot
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.title('Normalized Elo Difference and Z-Score Over Time (After April 2024)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
