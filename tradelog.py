from elodiff_graph import load_and_filter_data

# Get the filtered DataFrame
df_filtered = load_and_filter_data()

# Now you can use df_filtered in this file.
print(df_filtered.head())
print(df_filtered.columns)