import pandas as pd

# 1. Load and prepare data
df = pd.read_csv('output.csv', parse_dates=['DateTime'])
df.set_index('DateTime', inplace=True)

# 2. Resample to hourly means
hourly_original = df.resample('H').mean()

# 3. Create complete hourly index covering full date range
start_date = df.index.min().floor('H')  # Start of first hour with data
end_date = df.index.max().ceil('H')     # End of last hour with data
full_index = pd.date_range(start=start_date, end=end_date, freq='H')
hourly_complete = hourly_original.reindex(full_index)

hourly_complete.to_csv('hourly_complete.csv')  # Save for reference

# 4. Detect and display missing hours
missing_hours = hourly_complete[hourly_complete['Global_active_power'].isna()]
print(f"Missing hours detected: {len(missing_hours)}")
print(missing_hours.index.to_list())  # Show timestamps of missing hours

# 5. Fill missing hours with previous week's data
for idx in missing_hours.index:
    prev_week = idx - pd.Timedelta(weeks=1)
    if prev_week in hourly_original.index:
        hourly_complete.loc[idx, 'Global_active_power'] = hourly_original.loc[prev_week, 'Global_active_power']

# 6. Save final data
hourly_complete.reset_index().rename(columns={'index':'DateTime'}).to_csv('processed_data.csv', index=False)