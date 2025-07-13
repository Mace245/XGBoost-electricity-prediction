import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from lib import algo 

from scipy.stats import zscore
from collections import namedtuple
import openmeteo_requests
import requests_cache
from retry_requests import retry

def temp_fetch(start_date, end_date, latitude:float, longitude:float, historical:bool):
	cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

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
	response = responses[0]

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
	return hourly_dataframe

def fetch_elec_temp():
    electricity_data = pd.read_csv('Data/processed_hourly_Wh_data.csv', parse_dates=['DateTime'])
    electricity_data.set_index('DateTime', inplace=True)
    # electricity_data = electricity_data.drop(columns='Global_active_power')
    
    start_date = electricity_data.index.min().strftime('%Y-%m-%d')
    end_date = electricity_data.index.max().strftime('%Y-%m-%d')
    electricity_data = electricity_data.tz_localize('Asia/Kuala_Lumpur')
    print(start_date, end_date)

    latitude = 3.1219145808473048
    longitude = 101.65699508075299

    temperature_data = temp_fetch(
        start_date=start_date,
        end_date=end_date,
        latitude=latitude,
        longitude=longitude,
        historical=True
    )
    
    return electricity_data, temperature_data

from collections import namedtuple 

def get_all_data_from_db_for_training(db_session, energy_temp_reading_model, 
                                      output_df_target_col: str, output_df_temp_col: str,
                                      model_actual_target_attr: str, model_actual_temp_attr: str):
    print("Fetching all data from database for training...")
    all_readings_query = db_session.query(energy_temp_reading_model).order_by(energy_temp_reading_model.timestamp_utc).all()

    if not all_readings_query:
        print("No data in the database to train on.")
        return pd.DataFrame()

    df_data = []
    for r in all_readings_query:
        df_data.append({
            'DateTime': pd.to_datetime(r.timestamp_utc, utc=True),
            output_df_target_col: getattr(r, model_actual_target_attr, None),
            output_df_temp_col: getattr(r, model_actual_temp_attr, None)
        })

    df = pd.DataFrame(df_data)
    if df.empty:
        return df

    df = df.set_index('DateTime')
    print(f"Fetched {len(df)} records from DB. TZ: {df.index.tz}")
    return df

def handle_outliers(data, column='kWh', threshold=3):
    z_scores = zscore(data[column])
    print('zscore', data[(np.abs(z_scores) > threshold)])
    return data[(np.abs(z_scores) < threshold)]

def prepare_data(electricity_data:pd.DataFrame, temperature_data:pd.DataFrame):
    merged_data = electricity_data.merge(temperature_data, how='left', left_index=True, right_index=True)
    print(merged_data)
    merged_data = merged_data.ffill().dropna()
    return merged_data

def create_training_data(features:list[str], target:str):
    training_data = namedtuple("training_data", ['X', 'y'])
    electricity, temperature = fetch_elec_temp()
    merged_data = prepare_data(electricity, temperature)

    merged_data = algo.create_time_features(merged_data)
    merged_data = algo.create_lagged_features(merged_data)
    merged_data.to_csv('training_data.csv', index=True)

    X = merged_data[features]
    y = merged_data[target]
    return training_data(X=X, y=y)


def visualize(model, features:list[str], actual_values:pd.Series, forecast_values:pd.Series, period_label:str):
    comparison_df = pd.DataFrame({'Actual': actual_values, 'Forecast': forecast_values}).dropna()

    if comparison_df.empty:
        print(f"Warning: No overlapping data for period '{period_label}'.")
        return

    actuals_aligned = comparison_df['Actual']
    forecast_aligned = comparison_df['Forecast']
    
    rmse = np.sqrt(mean_squared_error(actuals_aligned, forecast_aligned))

    plt.figure(figsize=(6, 3))
    plt.plot(actuals_aligned.index, actuals_aligned, label='Actual')
    plt.plot(forecast_aligned.index, forecast_aligned, label='Forecast', linestyle='--')
    plt.title(f"{period_label} Comparison (RMSE: {rmse:.2f})")
    plt.xlabel("DateTime")
    plt.ylabel("Wh")
    plt.legend()
    plt.show()

def visualize_dms_forecast(
    dms_forecast_series: pd.Series,   
    actual_values_series: pd.Series, 
    period_label: str,
):
    comparison_df = pd.DataFrame({'Actual': actual_values_series, 'Forecast': dms_forecast_series}).dropna()

    actuals_aligned = comparison_df['Actual']
    forecast_aligned = comparison_df['Forecast']

    mse = mean_squared_error(actuals_aligned, forecast_aligned)
    mae = mean_absolute_error(actuals_aligned, forecast_aligned)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actuals_aligned - forecast_aligned) / actuals_aligned)) * 100 if np.all(actuals_aligned != 0) else float('inf')

    print(f"\n--- Evaluation for: {period_label} ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    plt.figure(figsize=(6, 3))
    plt.plot(actuals_aligned.index, actuals_aligned, label='Actual Values', marker='.', linestyle='-')
    plt.plot(forecast_aligned.index, forecast_aligned, label='DMS Forecast', marker='.', linestyle='--')
    plt.title(f"{period_label} Comparison\nRMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")
    plt.xlabel("DateTime")
    plt.ylabel("Wh") 
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()