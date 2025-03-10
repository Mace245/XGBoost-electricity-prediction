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
# print(missing_hours.index.to_list())  # Show timestamps of missing hours

count_filled = 0
count_total = 0
count_prev_week = 0
count_next_week = 0

# 5. Fill missing hours with previous week's data
for idx in missing_hours.index:
    filled_value = None  # to store the candidate replacement value
    source = None       # to track which date provided the replacement

    # Check the previous week value first
    prev_week = idx - pd.Timedelta(weeks=1)
    if prev_week in hourly_original.index:
        prev_value = hourly_original.loc[prev_week, 'Global_active_power']
        if not pd.isna(prev_value) and prev_value != 0:
            filled_value = prev_value
            source = prev_week
            count_prev_week += 1

    # If previous week's value is NaN or not available, check the next week
    if filled_value is None:
        next_week = idx + pd.Timedelta(weeks=1)
        if next_week in hourly_original.index:
            next_value = hourly_original.loc[next_week, 'Global_active_power']
            if not pd.isna(next_value) and next_value != 0:
                filled_value = next_value
                count_next_week += 1

    # If a valid replacement was found, fill the missing value
    if filled_value is not None:
        print(f"Filling missing hour at {idx} (current value: {hourly_complete.loc[idx, 'Global_active_power']}) "
              f"with data from {source} (value: {filled_value})")
        hourly_complete.loc[idx, 'Global_active_power'] = filled_value
        count_filled += 1

    count_total += 1
    
print(f"Missing hours filled: {count_filled}/{count_total}, with {count_prev_week} from previous week and {count_next_week} from next week")

# 6. Save final data
hourly_complete.reset_index().rename(columns={'index':'DateTime'}).to_csv('processed_hourly_data.csv', index=False)