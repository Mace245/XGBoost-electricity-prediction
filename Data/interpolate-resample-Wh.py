import pandas as pd
import matplotlib.pyplot as plt

# 1. Load and prepare data
df = pd.read_csv('output.csv', parse_dates=['DateTime'])
df.set_index('DateTime', inplace=True)

# 2. Resample to minute frequency and fill missing values with 0
print("Filling missing minutes...")
minute_data = df.resample('T').mean().fillna(0)

minute_data.to_csv('minute_data.csv', index=True)  # Save for reference

# 3. Calculate Wh for each minute (1 minute = 1/60 hours)
minute_data['TimeDiff'] = 1/60  # Fixed time difference for minutes
minute_data['Wh'] = minute_data['Global_active_power'] * minute_data['TimeDiff']

# 4. Aggregate to hourly data (mean power and total Wh)
hourly_original = minute_data.resample('H').agg({
    'Global_active_power': 'mean',
    'Wh': 'sum'
})

# 5. Create complete hourly index covering full date range
start_date = hourly_original.index.min().floor('h')
end_date = hourly_original.index.max().ceil('h') - pd.Timedelta(hours=0)
full_index = pd.date_range(start=start_date, end=end_date, freq='h')
hourly_complete = hourly_original.reindex(full_index)

hourly_complete.to_csv('hourly_indexed.csv')

# 6. Detect and display missing hours (now based on Wh)
missing_hours = hourly_complete[hourly_complete['Wh'] == 0]
print(f"Missing hours detected: {len(missing_hours)}")

count_filled = 0
count_total = 0
count_prev_week = 0
count_next_week = 0

# 7. Fill missing hours with previous week's data (both power and Wh)
for idx in missing_hours.index:
    filled_value = None
    source = None

    # Check previous weeks' data
    for weeks in [1, 2, 3]:
        prev_week = idx - pd.Timedelta(weeks=weeks)
        if prev_week in hourly_original.index:
            prev_value = hourly_original.loc[prev_week, 'Wh']
            if prev_value != 0:
                hourly_complete.loc[idx] = hourly_original.loc[prev_week]
                count_prev_week += 1
                filled_value = prev_value
                break

    # If not found, check following weeks
    if filled_value is None:
        for weeks in [1, 2, 3]:
            next_week = idx + pd.Timedelta(weeks=weeks)
            if next_week in hourly_original.index:
                next_value = hourly_original.loc[next_week, 'Wh']
                if next_value != 0:
                    hourly_complete.loc[idx] = hourly_original.loc[next_week]
                    count_next_week += 1
                    filled_value = next_value
                    break

    if filled_value is not None:
        count_filled += 1
    count_total += 1

print(f"Missing hours filled: {count_filled}/{count_total}, with {count_prev_week} from previous weeks and {count_next_week} from next weeks")

# 8. Save final data with Wh
hourly_complete.reset_index().rename(columns={'index':'DateTime'}).to_csv('processed_hourly_Wh_data.csv', index=False)

# Define the start and end of the one-week period
start_date = hourly_complete.index.min()
# end_date = start_date + pd.Timedelta(weeks=1)
end_date = hourly_complete.index.max()

# Subset the data for the chosen week
hourly_original_week = hourly_original.loc[start_date:end_date]
hourly_complete_week = hourly_complete.loc[start_date:end_date]
missing_hours_week = missing_hours.loc[start_date:end_date]

# 9. Plot results with Wh values
plt.figure(figsize=(14, 7))
plt.plot(hourly_original_week.index, hourly_original['Wh'], label='Original Data', alpha=0.5)
plt.plot(hourly_complete_week.index, hourly_complete['Wh'], label='Complete Data', alpha=0.8)
plt.scatter(missing_hours_week.index, hourly_complete_week.loc[missing_hours_week.index, 'Global_active_power'],
            color='red', label='Imputed Values')
plt.xlabel("Time")
plt.ylabel("Energy Consumption (Wh)")
plt.title("Energy Consumption with Missing Data Imputation")
plt.legend()
plt.show()
