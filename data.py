import pandas as pd
import numpy as np
from scipy.stats import zscore
import temp_api

# DATASET USE ACTUALLY KWH, NOT KW
# EXPLAIN SEASONAL DECOMPOSITION
# EXPLAIN LAGGED FEATURES

def fetch_elec_temp():
    electricity_data = pd.read_csv('Data/processed_hourly_data.csv', parse_dates=['DateTime'])
    electricity_data.set_index('DateTime', inplace=True)
    
    # Load temperature data
    # temperature_data = pd.read_csv('open-meteo-unix 20nov-22jan.csv')
    # temperature_data['time'] = pd.to_datetime(temperature_data['time'], unit='s')
    # temperature_data = temperature_data.set_index('time')[['temperature']]

    # Compute the formatted start and end dates
    start_date = electricity_data.index.min().strftime('%Y-%m-%d')
    end_date = electricity_data.index.max().strftime('%Y-%m-%d')
    electricity_data = electricity_data.tz_localize('Asia/Kuala_Lumpur')
    print(start_date, end_date)

    # Define the location coordinates
    latitude = 14.5833
    longitude = 121

    # Call the function with clearly named parameters
    temperature_data = temp_api.temp_fetch_historical(
        start_date=start_date,
        end_date=end_date,
        latitude=latitude,
        longitude=longitude
    )
    
    return electricity_data, temperature_data

def handle_outliers(data, column='Global_active_power', threshold=3):
    """Remove outliers using Z-score"""
    z_scores = zscore(data[column])
    return data[(np.abs(z_scores) < threshold)]

def prepare_data(electricity_data, temperature_data):
    # Resample electricity data to hourly
    # electricity_hourly = electricity_data.resample('h').mean()
    # electricity_hourly = electricity_hourly.head(1536) # temp for invalid merge
    # print(electricity_hourly.info(), temperature_data.info())

    print(electricity_data, temperature_data)
    
    # Merge electricity and temperature data
    merged_data = electricity_data.merge(temperature_data, how='left', left_index=True, right_index=True)
    print(merged_data)
    
    # Handle missing values and outliers
    merged_data = merged_data.ffill().dropna()
    merged_data = handle_outliers(merged_data)
    
    return merged_data

