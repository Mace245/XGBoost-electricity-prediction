import pandas as pd
import numpy as np
from scipy.stats import zscore

# DATASET USE ACTUALLY KWH, NOT KW
# EXPLAIN SEASONAL DECOMPOSITION
# EXPLAIN LAGGED FEATURES

def fetch_elec_temp():
    electricity_data = pd.read_csv('household_power_consumption.csv')
    
    # Convert datetime and handle NaNs
    electricity_data['DateTime'] = pd.to_datetime(
        electricity_data['Date'] + ' ' + electricity_data['Time'], 
        format='%d/%m/%Y %H:%M:%S'
    )
    electricity_data = electricity_data.set_index('DateTime')
    electricity_data = electricity_data[['Global_active_power']]
    electricity_data['Global_active_power'] = pd.to_numeric(electricity_data['Global_active_power'], errors='coerce')
    
    # Load temperature data
    temperature_data = pd.read_csv('open-meteo-unix 20nov-22jan.csv')
    temperature_data['time'] = pd.to_datetime(temperature_data['time'], unit='s')
    temperature_data = temperature_data.set_index('time')[['temperature']]
    
    return electricity_data, temperature_data

def handle_outliers(data, column='Global_active_power', threshold=3):
    """Remove outliers using Z-score"""
    z_scores = zscore(data[column])
    return data[(np.abs(z_scores) < threshold)]

def prepare_data(electricity_data, temperature_data):
    # Resample electricity data to hourly
    electricity_hourly = electricity_data.resample('h').mean()
    electricity_hourly = electricity_hourly.head(1536) # temp for invalid merge
    # print(electricity_hourly.info(), temperature_data.info())
    
    # Merge electricity and temperature data
    merged_data = electricity_hourly
    merged_data['temperature'] = temperature_data['temperature'].values
    # print(merged_data)
    
    # Handle missing values and outliers
    merged_data = merged_data.ffill().dropna()
    merged_data = handle_outliers(merged_data)
    
    return merged_data