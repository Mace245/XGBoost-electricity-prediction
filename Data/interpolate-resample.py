import pandas as pd
import matplotlib.pyplot as plt

# 1. Load and prepare data
df = pd.read_csv('output.csv', parse_dates=['DateTime'])
df.set_index('DateTime', inplace=True)

# 2. Resample to hourly means
hourly_original = df.resample('H').mean()
hourly_original['Global_active_power'] = hourly_original['Global_active_power'].fillna(0)

# 3. Create complete hourly index covering full date range
start_date = df.index.min().floor('H')  # Start of first hour with data
end_date = df.index.max().ceil('H')     # End of last hour with data
full_index = pd.date_range(start=start_date, end=end_date, freq='H')
hourly_complete = hourly_original.reindex(full_index)

hourly_complete.to_csv('hourly_indexed.csv')  # Save for reference

# 4. Detect and display missing hours
missing_hours = hourly_complete[hourly_complete['Global_active_power'] == 0]
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
    
    if filled_value is None:
        prev_2weeks = idx - pd.Timedelta(weeks=2)
        if prev_2weeks in hourly_original.index:
            prev_2weeks_value = hourly_original.loc[prev_2weeks, 'Global_active_power']
            if not pd.isna(prev_2weeks_value) and prev_2weeks_value != 0:
                filled_value = prev_2weeks_value
                count_prev_week += 1

    if filled_value is None:
        next_2weeks = idx + pd.Timedelta(weeks=2)
        if next_2weeks in hourly_original.index:
            next_2weeks_value = hourly_original.loc[next_2weeks, 'Global_active_power']
            if not pd.isna(next_2weeks_value) and next_2weeks_value != 0:
                filled_value = next_2weeks_value
                count_next_week += 1

    if filled_value is None:
        prev_3weeks = idx - pd.Timedelta(weeks=3)
        if prev_3weeks in hourly_original.index:
            prev_3weeks_value = hourly_original.loc[prev_3weeks, 'Global_active_power']
            if not pd.isna(prev_3weeks_value) and prev_3weeks_value != 0:
                filled_value = prev_3weeks_value
                count_prev_week += 1

    if filled_value is None:
        next_3weeks = idx + pd.Timedelta(weeks=3)
        if next_3weeks in hourly_original.index:
            next_3weeks_value = hourly_original.loc[next_3weeks, 'Global_active_power']
            if not pd.isna(next_3weeks_value) and next_3weeks_value != 0:
                filled_value = next_3weeks_value
                count_next_week += 1

    # If a valid replacement was found, fill the missing value
    if filled_value is not None:
        print(f"Filling missing hour at {idx} (current value: {hourly_complete.loc[idx, 'Global_active_power']}) "
              f"with data from {source} (value: {filled_value})")
        hourly_complete.loc[idx, 'Global_active_power'] = filled_value
        count_filled += 1

    count_total += 1
    
print(f"Missing hours filled: {count_filled}/{count_total}, with {count_prev_week} from previous week and {count_next_week} from next week")
missing_hours_after = hourly_complete[hourly_complete['Global_active_power'].isna()]
print(f"Missing hours after imputation: {len(missing_hours_after)}")


# 6. Save final data
hourly_complete.reset_index().rename(columns={'index':'DateTime'}).to_csv('processed_hourly_data.csv', index=False)

plt.figure(figsize=(14, 7))
plt.plot(hourly_original.index, hourly_original['Global_active_power'], label='Original Data', alpha=0.5)
plt.plot(hourly_complete.index, hourly_complete['Global_active_power'], label='Complete Data', alpha=0.8)
plt.scatter(missing_hours.index, hourly_complete.loc[missing_hours.index, 'Global_active_power'],
            color='red', label='Imputed Values')
plt.xlabel("Time")
plt.ylabel("Global Active Power")
plt.title("Time Series Data with Imputed Values")
plt.legend()
plt.show()
