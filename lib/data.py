import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from collections import namedtuple
import algo

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

def temp_fetch(start_date, end_date, latitude:float, longitude:float, historical:bool):
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	if historical == True:
		url = "https://archive-api.open-meteo.com/v1/archive"
	else:
		url = "https://api.open-meteo.com/v1/forecast"
	params = {
		"latitude": latitude,
		"longitude": longitude,
		"hourly": "temperature_2m",
		"timezone": "GMT+8",
		"start_date": start_date,
		"end_date": end_date
	}
	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	# print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	# print(f"Elevation {response.Elevation()} m asl")
	# print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
	# print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

	hourly_data = {"DateTime": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}

	hourly_data["temperature"] = hourly_temperature_2m

	hourly_dataframe = pd.DataFrame(data = hourly_data)
	hourly_dataframe.set_index('DateTime', inplace=True)
	print(f"Start Date: {start_date}")
	print(f"End Date: {end_date}")
	# print(hourly_dataframe)
	return hourly_dataframe

# temp_fetch_historical("2025-01-02", "2025-02-13", 14.5833, 121)

# DATASET USE ACTUALLY KWH, NOT KW
# EXPLAIN SEASONAL DECOMPOSITION
# EXPLAIN LAGGED FEATURES

def fetch_elec_temp():
    electricity_data = pd.read_csv('Data/processed_hourly_Wh_data.csv', parse_dates=['DateTime'])
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
    temperature_data = temp_fetch(
        start_date=start_date,
        end_date=end_date,
        latitude=latitude,
        longitude=longitude,
        historical=True
    )
    
    return electricity_data, temperature_data

def handle_outliers(data, column='Global_active_power', threshold=3):
    """Remove outliers using Z-score"""
    z_scores = zscore(data[column])
    return data[(np.abs(z_scores) < threshold)]

def prepare_data(electricity_data:pd.DataFrame, temperature_data:pd.DataFrame):
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

def create_training_data(features:list[str], target:str):
    training_data = namedtuple("training_data", ['X', 'y'])
    # Load data
    electricity, temperature = fetch_elec_temp()
    merged_data = prepare_data(electricity, temperature)

    # Feature engineering pipeline
    merged_data = algo.create_time_features(merged_data)
    merged_data = algo.create_lagged_features(merged_data)
    merged_data = algo.add_seasonal_components(merged_data)

    X = merged_data[features]
    y = merged_data[target]
    merged_data.to_csv('training_data.csv', index=True)
    return training_data(X=X, y=y)

def visualize(model, features:list[str], target_var:str, forecast:pd.Series, forecast_period:pd.Timedelta):
    mse = mean_squared_error(target_var.loc[target_var.index > target_var.index.max() - forecast_period], forecast)
    mae = mean_absolute_error(target_var.loc[target_var.index > target_var.index.max() - forecast_period], forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((target_var.loc[target_var.index > target_var.index.max() - forecast_period] - forecast) / target_var.loc[target_var.index > target_var.index.max() - forecast_period])) * 100

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    # Visualize
    # plt.figure(figsize=(12, 6))
    plt.plot(target_var.loc[target_var.index > target_var.index.max() - forecast_period], label='Historical')
    plt.plot(forecast, label='Forecast', linestyle='--')
    plt.title(forecast_period)
    plt.legend()
    plt.show()

    # get importance
    importance = model.feature_importances_
    print(importance)
    plt.bar(range(len(importance)), importance, align='center')
    plt.xticks(ticks=range(len(features)), labels=features, rotation=45)
    plt.ylabel("Importance Score")
    plt.title("Feature Importance")
    plt.show()

