import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from lib import algo # Keep other imports as they are

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
	# print(hourly_dataframe)
	return hourly_dataframe

# temp_fetch_historical("2025-01-02", "2025-02-13", 14.5833, 121)

# DATASET USE ACTUALLY KWH, NOT KW
# EXPLAIN SEASONAL DECOMPOSITION
# EXPLAIN LAGGED FEATURES

def fetch_elec_temp():
    electricity_data = pd.read_csv('Data/processed_hourly_Wh_data.csv', parse_dates=['DateTime'])
    electricity_data.set_index('DateTime', inplace=True)
    electricity_data = electricity_data.drop(columns='Global_active_power')
    
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

from collections import namedtuple # Ensure this is imported

# This function will be called by the training script/app
def get_all_data_from_db_for_training(db_session, energy_temp_reading_model, 
                                      output_df_target_col: str, output_df_temp_col: str,
                                      model_actual_target_attr: str, model_actual_temp_attr: str):
    """
    Fetches all data from the EnergyTempReading database and prepares it
    into a pandas DataFrame suitable for training.
    Returns a DataFrame with a DatetimeIndex (UTC) and columns for target and temperature.
    """
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
    # df = df.sort_index() # Ensure sorted by time

    # Handle potential missing values if any (though DB should be clean)
    # df[target_col_name] = df[target_col_name].astype(float)
    # df[temp_col_name] = df[temp_col_name].astype(float)
    # df = df.ffill().bfill() # Example imputation, adjust as needed

    print(f"Fetched {len(df)} records from DB. TZ: {df.index.tz}")
    return df

def handle_outliers(data, column='Wh', threshold=3):
    z_scores = zscore(data[column])
    print('zscore', data[(np.abs(z_scores) > threshold)])
    return data[(np.abs(z_scores) < threshold)]

def prepare_data(electricity_data:pd.DataFrame, temperature_data:pd.DataFrame):

    merged_data = electricity_data.merge(temperature_data, how='left', left_index=True, right_index=True)
    print(merged_data)
    
    merged_data = merged_data.ffill().dropna()
    # merged_data = handle_outliers(merged_data)
    return merged_data

def create_training_data(features:list[str], target:str):
    training_data = namedtuple("training_data", ['X', 'y'])
    electricity, temperature = fetch_elec_temp()
    merged_data = prepare_data(electricity, temperature)

    merged_data = algo.create_time_features(merged_data)
    merged_data = algo.create_lagged_features(merged_data)
    # merged_data = algo.add_seasonal_components(merged_data)

    X = merged_data[features]
    y = merged_data[target]
    # X.to_csv('training_dataX.csv', index=True)
    # y.to_csv('training_datay.csv', index=True)
    merged_data.to_csv('training_data.csv', index=True)
    return training_data(X=X, y=y)

def create_dms_training_data_for_horizon(
    merged_data_full: pd.DataFrame,
    features_list: list[str],
    target_col: str,
    horizon: int 
):
    """
    Prepares X and y for training a DMS model for a specific horizon.
    Features X will be from time 't'.
    Target y will be from time 't + horizon'.
    """
    data_with_features = merged_data_full.copy()

    # 1. Create time and lag features as usual.
    # These features will be relative to the original timestamp 't'.
    data_with_features = algo.create_time_features(data_with_features)
    # For DMS, lags are always based on actual past values relative to 't'
    data_with_features = algo.create_lagged_features(data_with_features, target_col=target_col)
    # Note: add_seasonal_components might be tricky with DMS target shifting if not careful.
    # For now, let's assume it's done before this function or not used for simplicity here.

    # 2. Create the shifted target variable for the DMS model
    # y_h is the value of the target_col 'horizon' steps into the future
    data_with_features[f'{target_col}_h{horizon}'] = data_with_features[target_col].shift(-horizon)

    # 3. Drop NaNs
    # NaNs will be created by create_lagged_features (at the beginning)
    # and by the target shift (at the end).
    data_cleaned = data_with_features.dropna(
        subset=features_list + [f'{target_col}_h{horizon}']
    )

    X = data_cleaned[features_list]
    y = data_cleaned[f'{target_col}_h{horizon}']

    DmsTrainingData = namedtuple("DmsTrainingData", ['X', 'y', 'full_data_with_target'])
    return DmsTrainingData(X=X, y=y, full_data_with_target=data_cleaned)


def create_dms_feature_set_for_prediction(
    history_df_slice: pd.DataFrame, # Slice of history ending at current time 't'
    features_list: list[str],
    target_col: str
):
    """
    Creates the feature set (a single row) for time 't' to be used for DMS prediction.
    Lags are calculated from history_df_slice.
    Time features are for time 't' (the last index of history_df_slice).
    """
    if history_df_slice.empty:
        raise ValueError("history_df_slice cannot be empty.")

    current_time_t = history_df_slice.index[-1]
    
    # Create features for the *last row* of history_df_slice
    # 1. Time features for 't'
    features_for_t_df = pd.DataFrame(index=[current_time_t])
    features_for_t_df = algo.create_time_features(features_for_t_df) # Creates features for index

    # 2. Lag features for 't' (lags of target_col from history_df_slice)
    for lag in [1, 24, 72, 168]: # Or get from your config
        lag_col_name = f'lag_{lag}'
        # Get the value from 'target_col' at 'current_time_t - lag'
        lag_timestamp = current_time_t - pd.Timedelta(hours=lag)
        if lag_timestamp in history_df_slice.index:
            features_for_t_df[lag_col_name] = history_df_slice.loc[lag_timestamp, target_col]
        else:
            features_for_t_df[lag_col_name] = np.nan # Or some other imputation

    # 3. Add other features like temperature (if it's part of features_list)
    # This assumes 'temperature' for time 't' is already in history_df_slice
    # or you have a way to get future temperatures if needed by any model.
    # For simplicity, let's assume 'temperature' at time 't' is available.
    if 'temperature' in features_list and 'temperature' in history_df_slice.columns:
        features_for_t_df['temperature'] = history_df_slice.loc[current_time_t, 'temperature']
    elif 'temperature' in features_list:
        # Handle missing future temperature for prediction if necessary
        # This example just puts NaN if not available.
        features_for_t_df['temperature'] = np.nan


    # Ensure all features are present and in correct order
    # This will select only the features in features_list and reorder them.
    # It will also add NaN for any feature in features_list not created above.
    final_feature_row = pd.DataFrame(columns=features_list, index=[current_time_t])
    for col in features_list:
        if col in features_for_t_df.columns:
            final_feature_row[col] = features_for_t_df[col]
    
    # Handle any NaNs (e.g., if a lag went too far back for the provided history_df_slice)
    final_feature_row = final_feature_row.fillna(method='ffill').bfill().fillna(0) # Basic imputation

    return final_feature_row

def visualize(model, features:list[str], actual_values:pd.Series, forecast_values:pd.Series, period_label:str):
    """
    Calculates metrics and visualizes the forecast against actual values for a specific period.

    Args:
        model: Trained XGBoost model object.
        features: List of feature names used by the model.
        actual_values: Pandas Series of the true target values for the evaluation period.
        forecast_values: Pandas Series of the predicted values for the evaluation period.
        period_label: String label for the period (e.g., "7-Day Verification", "Next 7 Days Forecast").
    """
    # Ensure indices align for comparison, drop any NaNs that might result
    comparison_df = pd.DataFrame({'Actual': actual_values, 'Forecast': forecast_values}).dropna()

    if comparison_df.empty:
        print(f"Warning: No overlapping data between actuals and forecast for period '{period_label}'. Cannot calculate metrics.")
        # Optionally plot forecast if available
        if not forecast_values.empty:
             plt.figure(figsize=(12, 6))
             plt.plot(forecast_values, label='Forecast Only', linestyle='--')
             plt.title(f"{period_label} (No Actuals for Comparison)")
             plt.legend()
             plt.show()
        return

    actuals_aligned = comparison_df['Actual']
    forecast_aligned = comparison_df['Forecast']

    mse = mean_squared_error(actuals_aligned, forecast_aligned)
    mae = mean_absolute_error(actuals_aligned, forecast_aligned)
    rmse = np.sqrt(mse)


    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Visualize
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

    # Calculate metrics
    mse = mean_squared_error(actuals_aligned, forecast_aligned)
    mae = mean_absolute_error(actuals_aligned, forecast_aligned)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actuals_aligned - forecast_aligned) / actuals_aligned)) * 100 if np.all(actuals_aligned != 0) else float('inf')


    print(f"\n--- Evaluation for: {period_label} ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


    # Visualize Forecast vs Actual
    plt.figure(figsize=(6, 3))
    plt.plot(actuals_aligned.index, actuals_aligned, label='Actual Values', marker='.', linestyle='-')
    plt.plot(forecast_aligned.index, forecast_aligned, label='DMS Forecast', marker='.', linestyle='--')
    plt.title(f"{period_label} Comparison\nRMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")
    plt.xlabel("DateTime")
    plt.ylabel("Wh") # Or your target variable name
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()